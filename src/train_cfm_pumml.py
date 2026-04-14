# train_cfm_pumml.py
# X (generated) = ch_neutral_lv flattened          (100,)
# Y (context)   = CNN embedding of 3 image channels (128,)
#
# The CNN encoder (PummlEncoder) is trained jointly with the CFM:
# gradients flow through the ODE loss into the encoder weights.
# This means the encoder learns to produce embeddings that are
# maximally useful for the flow, not just for image reconstruction.

import yaml
import time
import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchdiffeq import odeint

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

from create_cfm_model import build_cfm_model, save_cfm_model, resume_cfm_model
from modded_cfm import (
    ModelWrapper,
    MyTargetConditionalFlowMatcher,
    AlphaTConditionalFlowMatcher,
    MyAlphaTTargetConditionalFlowMatcher,
)
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    SchrodingerBridgeConditionalFlowMatcher,
)
from pumml_encoder import PummlEncoder, preprocess_images, preprocess_targets
from data_preprocessing_pumml import (
    PummlTrainPreprocessor,
    PummlTestPreprocessor,
    TARGET_DIM,
    CONTEXT_DIM,
)
from validation_pumml import validate_pumml



# Helpers  (mirror train_cfm.py)
def get_device(gpu):
    if gpu is not None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


def prepare_output_dirs(log_name):
    log_name = log_name or f"pumml_cfm_{int(time.time())}"
    log_dir  = f"./logs/{log_name}"
    save_dir = f"./checkpoints/{log_name}"
    os.makedirs(log_dir,  exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    return log_dir, save_dir


def build_flow_matcher(base_kwargs):
    sigma         = base_kwargs["cfm"]["sigma"]
    matching_type = base_kwargs["cfm"]["matching_type"]
    if matching_type == "Target":
        return MyTargetConditionalFlowMatcher(sigma=sigma)
    elif matching_type == "AlphaT":
        return AlphaTConditionalFlowMatcher(sigma=sigma, alpha=base_kwargs["cfm"]["alpha"])
    elif matching_type == "AlphaTTarget":
        return MyAlphaTTargetConditionalFlowMatcher(sigma=sigma, alpha=base_kwargs["cfm"]["alpha"])
    elif matching_type == "Default":
        return ConditionalFlowMatcher(sigma=sigma)
    elif matching_type == "ExactOptimal":
        return ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif matching_type == "SchrodingerBridge":
        return SchrodingerBridgeConditionalFlowMatcher(sigma=sigma)
    raise ValueError(f"Unknown matching_type: '{matching_type}'")


def build_scheduler(optimizer, name):
    if name == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=20, verbose=True
        )
    elif name == "StepLR":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5, verbose=True)
    print("No scheduler.")
    return None


def get_noise_dist(data_kwargs, base_kwargs):
    name = data_kwargs.get("noise_distribution", "gaussian")
    if name == "gaussian":
        return torch.randn
    if name == "uniform":
        return torch.rand
    raise ValueError(f"Unknown noise_distribution: '{name}'")



# Encode context on-the-fly (used during training when freeze_encoder=False)
def encode_batch_online(encoder, imgs_batch, device):
    """
    Run the CNN encoder on a raw image batch during the training loop.
    imgs_batch : (B, 3, 40, 40) float32 tensor already on CPU
    Returns    : (B, 128) float32 tensor on device
    """
    return encoder(imgs_batch.to(device))


# Sampling
def sample_from_model(model, encoder, X_test_tensor, imgs_test,
                      test_batch_size, noise_dist, device, base_kwargs, context_dim):
    model.eval()
    encoder.eval()

    # torchdiffeq uses float64 internally which MPS does not support.
    # Move model and encoder to CPU just for the ODE solve.
    ode_device = torch.device("cpu")
    model.to(ode_device)
    encoder.to(ode_device)

    sampler   = ModelWrapper(model, context_dim=context_dim)
    timesteps = base_kwargs["cfm"]["timesteps"]
    t_span    = torch.linspace(0, 1, timesteps).to(ode_device)

    samples_list = []
    n_batches = (len(X_test_tensor) + test_batch_size - 1) // test_batch_size

    with torch.no_grad():
        with tqdm(total=n_batches, desc="Sampling", dynamic_ncols=True) as pbar:
            for i in range(0, len(X_test_tensor), test_batch_size):
                imgs_b = imgs_test[i : i + test_batch_size].to(ode_device)
                Yb     = encoder(imgs_b)
                x0     = noise_dist(len(Yb), X_test_tensor.shape[1]).to(ode_device)
                ic     = torch.cat([x0, Yb], dim=-1)

                while True:
                    try:
                        s = odeint(
                            sampler, ic, t_span,
                            atol=1e-5, rtol=1e-5, method="dopri5"
                        )[timesteps - 1, :, : X_test_tensor.shape[1]]
                        break
                    except AssertionError:
                        print("ODE assertion error, retrying.")

                samples_list.append(s.cpu().numpy())
                pbar.update(1)

    # Move back to training device
    model.to(device)
    encoder.to(device)

    return np.concatenate(samples_list, axis=0)



# Main training function

def train(input_dim, context_dim, gpu, train_kwargs, data_kwargs, base_kwargs):
    device   = get_device(gpu)
    log_dir, save_dir = prepare_output_dirs(train_kwargs["log_name"])
    writer   = SummaryWriter(log_dir=log_dir)

    freeze_encoder = train_kwargs.get("freeze_encoder", False)

    # encoder
    encoder = PummlEncoder(
        embed_dim=context_dim,   # must equal context_dim in YAML
        dropout=base_kwargs.get("encoder_dropout", 0.0),
    ).to(device)

    # CFM model 
    model = build_cfm_model(input_dim, context_dim, base_kwargs).to(device)
    fm    = build_flow_matcher(base_kwargs)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    enc_params   = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"CFM parameters    : {total_params:,}")
    print(f"Encoder parameters: {enc_params:,}")
    writer.add_scalar("cfm_params",     total_params, 0)
    writer.add_scalar("encoder_params", enc_params,   0)

    lr        = train_kwargs["lr"]
    epochs    = train_kwargs["epochs"]
    batch_size      = data_kwargs["batch_size"]
    test_batch_size = data_kwargs["test_batch_size"]

    # Optimizer: include encoder params unless frozen
    if freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad = False
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        print("Encoder is FROZEN.")
    else:
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(encoder.parameters()), lr=lr
        )
        print("Encoder is trained jointly with CFM.")

    scheduler = build_scheduler(optimizer, train_kwargs.get("scheduler"))

    # Resume
    start_epoch = 0
    latest = os.path.join(save_dir, "checkpoint-latest.pt")
    if train_kwargs.get("resume", False) and os.path.exists(latest):
        model, start_epoch, lr = resume_cfm_model(save_dir, "checkpoint-latest.pt")
        model = model.to(device)
        print(f"Resumed from epoch {start_epoch}.")

    #  data 
    # Pre-encode training images (frozen encoder path) OR load raw images
    # (joint training path where we re-encode each batch online).
    npz_path = data_kwargs["dataset_path"]
    N_train  = data_kwargs["N_train"]
    N_test   = data_kwargs["N_test"]

    raw = dict(np.load(npz_path))

    # Always pre-load images as tensors so we can slice them in the loop.
    print("Loading training images...")
    imgs_train = preprocess_images(raw, 0,       N_train)          # (N, 3, 40, 40)
    X_train_np = preprocess_targets(raw, 0,       N_train).numpy() # (N, 100)

    print("Loading test images...")
    imgs_test  = preprocess_images(raw, N_train, N_train + N_test) # (N, 3, 40, 40)
    X_test_np  = preprocess_targets(raw, N_train, N_train + N_test).numpy()

    # Physical (raw pT) copy for validation
    X_test_physical = np.clip(X_test_np.copy(), 0.0, None)
    imgs_test_physical = imgs_test.numpy()

    # Apply log1p to targets
    if data_kwargs.get("log1p", True):
        X_train_np = np.log1p(np.clip(X_train_np, 0.0, None))
        X_test_np  = np.log1p(np.clip(X_test_np,  0.0, None))

    # Fit / load StandardScaler for X
    from sklearn.preprocessing import StandardScaler
    from data_preprocessing_pumml import MEAN_X_PATH, SCALE_X_PATH, _init_or_load_scaler_x
    if data_kwargs.get("standardize", True):
        scaler_x = _init_or_load_scaler_x(X_train_np)
        X_train_np = scaler_x.transform(X_train_np)
        X_test_np  = scaler_x.transform(X_test_np)
    else:
        scaler_x = None

    X_train_t = torch.tensor(X_train_np, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test_np,  dtype=torch.float32)

    noise_dist = get_noise_dist(data_kwargs, base_kwargs)

    print(f"Start epoch: {start_epoch}  End epoch: {epochs}")

    csv_path = os.path.join(save_dir, "losses.csv")
    with open(csv_path, "w") as f:
        f.write("epoch,train_loss,test_loss\n")


    # Training loop
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        if not freeze_encoder:
            encoder.train()

        print_lr    = optimizer.param_groups[0]["lr"]
        train_loss  = 0.0
        n_batches   = 0
        total_batches = (len(X_train_t) + batch_size - 1) // batch_size

        with tqdm(total=total_batches, desc=f"Epoch {epoch}", dynamic_ncols=True, ascii=True) as pbar:
            for i in range(0, len(X_train_t), batch_size):
                Xb    = X_train_t[i : i + batch_size].to(device)
                imgs_b = imgs_train[i : i + batch_size].to(device)

                # Encode context Y on the fly (gradients flow into encoder)
                Yb = encoder(imgs_b)   # (B, 128)

                optimizer.zero_grad()
                x0 = noise_dist(len(Xb), Xb.shape[1]).to(device)
                t, xt, ut = fm.sample_location_and_conditional_flow(x0, Xb)
                vt   = model(xt, context=Yb, flow_time=t[:, None])
                loss = torch.mean((vt - ut) ** 2)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                n_batches  += 1
                pbar.update(1)
                pbar.set_postfix({"loss": f"{loss.item():.5f}"})

        train_loss /= max(n_batches, 1)
        writer.add_scalar("loss",       train_loss, epoch)
        writer.add_scalar("lr",         print_lr,   epoch)

        # test loss
        model.eval()
        encoder.eval()
        test_loss = 0.0
        n_test_b  = 0
        with torch.no_grad():
            for i in range(0, len(X_test_t), test_batch_size):
                Xb     = X_test_t[i : i + test_batch_size].to(device)
                imgs_b = imgs_test[i : i + test_batch_size].to(device)
                Yb     = encoder(imgs_b)
                x0     = noise_dist(len(Xb), Xb.shape[1]).to(device)
                t, xt, ut = fm.sample_location_and_conditional_flow(x0, Xb)
                vt = model(xt, context=Yb, flow_time=t[:, None])
                test_loss += torch.mean((vt - ut) ** 2).item()
                n_test_b  += 1
        test_loss /= max(n_test_b, 1)
        writer.add_scalar("test_loss", test_loss, epoch)

        print(f"  lr={print_lr:.6f}  train={train_loss:.6f}  test={test_loss:.6f}")

        if scheduler is not None:
            if train_kwargs.get("scheduler") == "ReduceLROnPlateau":
                scheduler.step(train_loss)
            else:
                scheduler.step()

        with open(csv_path, "a") as f:
            f.write(f"{epoch},{train_loss},{test_loss}\n")

        # periodic sampling + validation
        if epoch % train_kwargs["eval_freq"] == 0:
            print("Sampling...")
            samples = sample_from_model(
                model=model, encoder=encoder,
                X_test_tensor=X_test_t, imgs_test=imgs_test,
                test_batch_size=test_batch_size,
                noise_dist=noise_dist, device=device,
                base_kwargs=base_kwargs, context_dim=context_dim,
            )

            # Back to physical pT
            if data_kwargs.get("standardize", True) and scaler_x is not None:
                samples = scaler_x.inverse_transform(samples)
            if data_kwargs.get("log1p", True):
                samples = np.expm1(np.clip(samples, 0.0, None))
            samples = np.clip(samples, 0.0, None)

            np.save(os.path.join(save_dir, "samples_pumml.npy"),         samples)
            np.save(os.path.join(save_dir, "X_test_physical_pumml.npy"), X_test_physical)
            np.save(os.path.join(save_dir, "imgs_test_pumml.npy"),        imgs_test_physical)

            validate_pumml(
                samples=samples,
                X_test_physical=X_test_physical,
                imgs_test=imgs_test_physical,
                save_dir=save_dir,
                epoch=epoch,
                writer=writer,
            )

        # periodic checkpoint
        if epoch % train_kwargs["save_freq"] == 0:
            save_cfm_model(
                model=model, epoch=epoch, lr=print_lr,
                name=train_kwargs["log_name"],
                input_dim=input_dim, context_dim=context_dim,
                base_kwargs=base_kwargs, path=save_dir,
            )
            # Also save encoder separately
            torch.save(
                {"epoch": epoch, "encoder_state_dict": encoder.state_dict()},
                os.path.join(save_dir, "encoder-latest.pt"),
            )
            print("Checkpoint saved.")

    writer.close()



# Entry point
if __name__ == "__main__":
    args        = sys.argv
    config_path = ("../configs/" + args[1]) if len(args) > 1 else "../configs/CRT_pumml_cnn.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    train(
        input_dim    = config["input_dim"],
        context_dim  = config["context_dim"],
        gpu          = config["gpu"],
        train_kwargs = config["train_kwargs"],
        data_kwargs  = config["data_kwargs"],
        base_kwargs  = config["base_kwargs"],
    )