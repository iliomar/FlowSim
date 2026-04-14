# train_cfm.py
# Train a Conditional Flow Matching (CFM) model on jet data.
#
# This script:
# 1. loads the YAML configuration,
# 2. builds the CFM model,
# 3. preprocesses training and test datasets,
# 4. trains the model,
# 5. periodically samples from the learned flow,
# 6. validates generated samples against the test set,
# 7. saves checkpoints and loss history.
#
# Notes for the current dataset setup:
# - X has 16 reconstructed features.
# - Y is the context vector.
# - If flavour_ohe=True, Y after preprocessing has:
#   [pt_gen, eta_gen, phi_gen, m_gen, muon_pT, jetR, jetArea, flavour_one_hot...]
# - With flavour categories [0, 1, 2, 3, 4, 5, 6, 21], the final context_dim is 15.
#
# Make sure your YAML config uses:
#   input_dim: 16
#   context_dim: 15
# when flavour_ohe is enabled.

import yaml
import time
import os 
import sys

import numpy as np
import torch
from tqdm import tqdm

# add the data directory to the Python path so the preprocessing module can be imported.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

# project imports
from create_cfm_model import build_cfm_model, save_cfm_model, resume_cfm_model
from modded_cfm import ModelWrapper
from validation import validate
from data_preprocessing import TrainDataPreprocessor, TestDataPreprocessor


from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter

from torchdiffeq import odeint

# import torchsde
# from torchdyn.core import NeuralODE
from rqdm import tqdm



# Conditional Flow Matching classes.
from torchcfm.conditional_flow_matching import *
from modded_cfm import (
    MyTargetConditionalFlowMatcher,
    AlphaTConditionalFlowMatcher,
    MyAlphaTTargetConditionalFlowMatcher,
)

# Fixed flavour category ordering used by preprocessing.
# This must match the one-hot encoding order used in data_preprocessing.py.
FLAVOUR_CATEGORIES = np.array([0, 1, 2, 3, 4, 5, 6, 21])

# checking if gpu is available (if not use cpu) - *Added*
def get_device(gpu):
    if gpu is not None:
        device = torch.device("mps" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    print("Using device:", device)
    return device

# create logging and checkpoints dir - *Added*
def prepare_output_dirs(log_name):
    """
    Parameters
    ----------
    log_name : str or None
        Name of the logging/checkpoint directory. If None, a timestamp is used.
    """
    if log_name is not None:
        log_dir = f"./logs/{log_name}"
        save_dir = f"./checkpoints/{log_name}"
    else:
        timestamp = int(time.time())
        log_dir = f"./logs/time-{timestamp}"
        save_dir = f"./checkpoints/time-{timestamp}"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    return log_dir, save_dir

# construct the chosen Conditional Flow Matching object - *Added*
def build_flow_matcher(base_kwargs):
    """
    Parameters
    ----------
    base_kwargs : dict
        Model configuration dictionary from the YAML config.

    Returns
    -------
    object
        An initialized flow matcher.
    """
    sigma = base_kwargs["cfm"]["sigma"]
    matching_type = base_kwargs["cfm"]["matching_type"]

    if matching_type == "Target":
        fm = MyTargetConditionalFlowMatcher(sigma=sigma)
    elif matching_type == "AlphaT":
        fm = AlphaTConditionalFlowMatcher(
            sigma=sigma,
            alpha=base_kwargs["cfm"]["alpha"],
        )
    elif matching_type == "AlphaTTarget":
        fm = MyAlphaTTargetConditionalFlowMatcher(
            sigma=sigma,
            alpha=base_kwargs["cfm"]["alpha"],
        )
    elif matching_type == "Default":
        fm = ConditionalFlowMatcher(sigma=sigma)
    elif matching_type == "ExactOptimal":
        fm = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif matching_type == "SchrodingerBridge":
        fm = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma)
    else:
        raise ValueError(f"Matching type '{matching_type}' not found.")

    return fm


def build_scheduler(optimizer, scheduler_name):
    """
    Create the learning-rate scheduler.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer used for training.
    scheduler_name : str
        Scheduler name from the config.

    Returns
    -------
    torch.optim.lr_scheduler._LRScheduler or None
        The scheduler, or None if no valid scheduler is selected.
    """
    if scheduler_name == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=30,
            verbose=True,
        )
    elif scheduler_name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.5,
            verbose=True,
        )
    else:
        scheduler = None
        print("Scheduler not found, proceeding without it.")

    return scheduler


def get_noise_distribution(data_kwargs, base_kwargs):
    """
    Select the latent noise distribution used for flow matching.

    Parameters
    ----------
    data_kwargs : dict
        Data-related configuration.
    base_kwargs : dict
        Model-related configuration.

    Returns
    -------
    callable
        A function like torch.randn or torch.rand.
    """
    noise_name = data_kwargs["noise_distribution"]
    matching_type = base_kwargs["cfm"]["matching_type"]

    if noise_name == "gaussian":
        print("Gaussian noise")
        return torch.randn

    if noise_name == "uniform" and matching_type in ["Default", "AlphaT"]:
        print("Uniform noise")
        return torch.rand

    raise ValueError(
        "Noise distribution not found for this combination of matching type "
        "and noise distribution."
    )


def inverse_transform_test_data(X_test, Y_test, test_dataset, data_kwargs):
    """
    Convert standardized test data back to physical units for validation.

    Parameters
    ----------
    X_test : np.ndarray
        Standardized reconstructed test features.
    Y_test : np.ndarray
        Standardized context test features.
    test_dataset : TestDataPreprocessor
        The fitted test preprocessor object.
    data_kwargs : dict
        Data configuration.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Physical-space copies of X_test and Y_test.
    """
    X_test_cpu = X_test.copy()
    Y_test_cpu = Y_test.copy()

    if data_kwargs["standardize"]:
        X_test_cpu = test_dataset.scaler_x.inverse_transform(X_test_cpu)

        # When flavour OHE is enabled, only the first 7 Y columns are continuous.
        if data_kwargs["flavour_ohe"]:
            Y_test_cpu[:, 0:7] = test_dataset.scaler_y.inverse_transform(
                Y_test_cpu[:, 0:7]
            )
        else:
            Y_test_cpu = test_dataset.scaler_y.inverse_transform(Y_test_cpu)

    return X_test_cpu, Y_test_cpu


def reconstruct_physical_context(Y_test_cpu, flavour_ohe):
    """
    Reconstruct the compact physical context vector from the processed Y array.

    If flavour OHE is enabled, the compact physical context returned is:
    [pt_gen, eta_gen, phi_gen, m_gen, flavour, muon_pT, jetR, jetArea]

    Parameters
    ----------
    Y_test_cpu : np.ndarray
        Physical-space context array after inverse scaling.
    flavour_ohe : bool
        Whether one-hot flavour encoding was used.

    Returns
    -------
    np.ndarray
        Reconstructed compact physical context array.
    """
    if not flavour_ohe:
        return Y_test_cpu

    if Y_test_cpu.shape[1] <= 7:
        raise ValueError(
            "Expected one-hot flavour block in Y_test_cpu, but the shape is too small."
        )

    # Extract the one-hot flavour block.
    flavour_block = Y_test_cpu[:, 7:]

    # Convert one-hot entries back to their original flavour labels.
    flavour_indices = np.argmax(flavour_block, axis=1)
    flavour = FLAVOUR_CATEGORIES[flavour_indices]

    # Rebuild Y in compact physical form:
    # [pt_gen, eta_gen, phi_gen, m_gen, flavour, muon_pT, jetR, jetArea]
    Y_test_physical = np.hstack(
        (
            Y_test_cpu[:, :4],
            flavour.reshape(-1, 1),
            Y_test_cpu[:, 4:7],
        )
    )

    print("Reconstructed physical Y shape:", Y_test_physical.shape)
    return Y_test_physical


def round_integer_like_features(X):
    """
    Round the features that represent count-like quantities.

    X column layout:
    0:  btag
    1:  recoPt / pt_gen
    2:  recoPhi
    3:  recoEta
    4:  recoNConst
    5:  nef
    6:  nhf
    7:  cef
    8:  chf
    9:  qgl
    10: jetId
    11: ncharged
    12: nneutral
    13: ctag
    14: nSV
    15: recoMass / m_gen
    """
    X[:, 4] = np.round(X[:, 4])   # recoNConst
    X[:, 11] = np.round(X[:, 11]) # ncharged
    X[:, 12] = np.round(X[:, 12]) # nneutral
    X[:, 14] = np.round(X[:, 14]) # nSV
    return X


def compute_test_loss(model, fm, X_test, Y_test, test_batch_size, noise_dist, device):
    """
    Evaluate the model loss on the test set.

    Parameters
    ----------
    model : torch.nn.Module
        The CFM model.
    fm : object
        Flow matcher instance.
    X_test : torch.Tensor
        Test reconstructed features.
    Y_test : torch.Tensor
        Test context features.
    test_batch_size : int
        Batch size for evaluation.
    noise_dist : callable
        Noise distribution function.
    device : torch.device
        Active device.

    Returns
    -------
    float
        Mean test loss.
    """
    model.eval()
    test_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for i in range(0, len(X_test), test_batch_size):
            X_batch = X_test[i : i + test_batch_size]
            Y_batch = Y_test[i : i + test_batch_size]

            x0 = noise_dist(X_batch.shape[0], X_batch.shape[1]).to(device)
            t, xt, ut = fm.sample_location_and_conditional_flow(x0, X_batch)

            vt = model(xt, context=Y_batch, flow_time=t[:, None])
            loss = torch.mean((vt - ut) ** 2)

            test_loss += loss.item()
            total_batches += 1

    if total_batches == 0:
        raise ValueError("No test batches were processed. Check test dataset size.")

    return test_loss / total_batches


def sample_from_model(
    model,
    X_test,
    Y_test,
    test_batch_size,
    noise_dist,
    device,
    base_kwargs,
    context_dim,
):
    """
    Generate samples from the trained model by integrating the learned ODE.

    Parameters
    ----------
    model : torch.nn.Module
        Trained CFM model.
    X_test : torch.Tensor
        Test reconstructed features, used only for shapes.
    Y_test : torch.Tensor
        Test context features used for conditioning.
    test_batch_size : int
        Sampling batch size.
    noise_dist : callable
        Noise distribution.
    device : torch.device
        Active device.
    base_kwargs : dict
        Model configuration dictionary.
    context_dim : int
        Dimension of the context vector.

    Returns
    -------
    np.ndarray
        Generated samples in standardized model space.
    """
    model.eval()
    samples_list = []
    timesteps = base_kwargs["cfm"]["timesteps"]
    ode_backend = base_kwargs["cfm"]["ode_backend"]

    sampler = ModelWrapper(model, context_dim=context_dim)

    if ode_backend == "torchdyn":
        from torchdyn.core import NeuralODE

        node = NeuralODE(
            sampler,
            solver="dopri5",
            sensitivity="adjoint",
            atol=1e-5,
            rtol=1e-5,
        )

    t_span = torch.linspace(0, 1, timesteps).to(device)

    total_batches = len(X_test) // test_batch_size
    if len(X_test) % test_batch_size != 0:
        total_batches += 1

    with torch.no_grad():
        with tqdm(
            total=total_batches,
            desc="Sampling",
            dynamic_ncols=True,
        ) as pbar:
            for i in range(0, len(X_test), test_batch_size):
                Y_batch = Y_test[i : i + test_batch_size, :]

                # Retry if the ODE solver hits an internal assertion issue.
                while True:
                    try:
                        x0_sample = noise_dist(len(Y_batch), X_test.shape[1]).to(device)
                        initial_conditions = torch.cat([x0_sample, Y_batch], dim=-1)

                        if ode_backend == "torchdyn":
                            samples = node.trajectory(
                                initial_conditions,
                                t_span,
                            )[timesteps - 1, :, : X_test.shape[1]]
                        elif ode_backend == "torchdiffeq":
                            samples = odeint(
                                sampler,
                                initial_conditions,
                                t_span,
                                atol=1e-5,
                                rtol=1e-5,
                                method="dopri5",
                            )[timesteps - 1, :, : X_test.shape[1]]
                        else:
                            raise ValueError("ODE backend not found.")

                        break

                    except AssertionError:
                        print("Assertion error during ODE solve, retrying.")

                samples_list.append(samples.detach().cpu().numpy())
                pbar.update(1)

    samples = np.concatenate(samples_list, axis=0)
    samples = np.array(samples).reshape((-1, X_test.shape[1]))
    return samples


def clip_samples_to_test_range(samples, X_test_cpu):
    """
    Clip generated samples feature-by-feature to the physical range of the test set.

    Parameters
    ----------
    samples : np.ndarray
        Generated samples in physical space.
    X_test_cpu : np.ndarray
        Real test samples in physical space.

    Returns
    -------
    np.ndarray
        Clipped samples.
    """
    for i in range(samples.shape[1]):
        samples[:, i] = np.clip(
            samples[:, i],
            X_test_cpu[:, i].min(),
            X_test_cpu[:, i].max(),
        )
    return samples


def train(input_dim, context_dim, gpu, train_kwargs, data_kwargs, base_kwargs):
    """
    Main training function.

    Parameters
    ----------
    input_dim : int
        Dimension of the reconstructed feature vector X.
    context_dim : int
        Dimension of the conditioning vector Y.
    gpu : any
        GPU flag from the config.
    train_kwargs : dict
        Training hyperparameters.
    data_kwargs : dict
        Dataset and preprocessing configuration.
    base_kwargs : dict
        Model architecture and flow-matching configuration.
    """
    # Select device.
    device = get_device(gpu)

    # Prepare directories for TensorBoard logs and model checkpoints.
    log_dir, save_dir = prepare_output_dirs(train_kwargs["log_name"])
    writer = SummaryWriter(log_dir=log_dir)

    # Build the model and corresponding flow matcher.
    model = build_cfm_model(input_dim, context_dim, base_kwargs)
    fm = build_flow_matcher(base_kwargs)

    # Training hyperparameters.
    lr = train_kwargs["lr"]
    start_epoch = 0
    epochs = train_kwargs["epochs"]
    batch_size = data_kwargs["batch_size"]
    test_batch_size = data_kwargs["test_batch_size"]

    # Resume from checkpoint if requested or if a latest checkpoint exists.
    latest_checkpoint_path = os.path.join(save_dir, "checkpoint-latest.pt")
    if train_kwargs["resume_checkpoint"] is None and os.path.exists(latest_checkpoint_path):
        resume_checkpoint = "checkpoint-latest.pt"
    else:
        resume_checkpoint = train_kwargs["resume_checkpoint"]

    if resume_checkpoint is not None and train_kwargs["resume"] is True:
        model, start_epoch, lr = resume_cfm_model(save_dir, resume_checkpoint)
        print(f"Resumed from epoch: {start_epoch}")

    # Move the model to the selected device.
    model = model.to(device)

    # Print and log the number of trainable parameters.
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable parameters:", total_params)
    writer.add_scalar("total_params", total_params, 0)

    # Build optimizer and scheduler.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = build_scheduler(optimizer, train_kwargs["scheduler"])

    # Build training and test datasets using the same fitted scalers.
    train_dataset = TrainDataPreprocessor(data_kwargs)
    X_train_np, Y_train_np = train_dataset.get_dataset()

    test_dataset = TestDataPreprocessor(
        data_kwargs,
        scaler_x=train_dataset.scaler_x,
        scaler_y=train_dataset.scaler_y,
    )
    X_test_np, Y_test_np = test_dataset.get_dataset()

    # Convert training and test arrays to PyTorch tensors.
    X_train = torch.tensor(X_train_np).float().to(device)
    Y_train = torch.tensor(Y_train_np).float().to(device)
    X_test = torch.tensor(X_test_np).float().to(device)
    Y_test = torch.tensor(Y_test_np).float().to(device)

    # Prepare physical-space copies of the test set for validation plots and metrics.
    X_test_cpu, Y_test_cpu = inverse_transform_test_data(
        X_test_np,
        Y_test_np,
        test_dataset,
        data_kwargs,
    )
    Y_test_cpu = reconstruct_physical_context(
        Y_test_cpu,
        flavour_ohe=data_kwargs["flavour_ohe"],
    )
    X_test_cpu = round_integer_like_features(X_test_cpu)

    # Select the latent noise distribution.
    noise_dist = get_noise_distribution(data_kwargs, base_kwargs)

    print(f"Start epoch: {start_epoch} End epoch: {epochs}")

    train_history = []
    test_history = []

    # Main training loop.
    for epoch in range(start_epoch, epochs + 1):
        model.train()

        # Print current learning rate.
        for param_group in optimizer.param_groups:
            print_lr = param_group["lr"]
            print(f"Current lr is {print_lr:.8f}")

        train_loss = 0.0
        total_train_batches = len(X_train) // batch_size
        if len(X_train) % batch_size != 0:
            total_train_batches += 1

        # Loop over training batches.
        with tqdm(
            total=total_train_batches,
            desc="Training",
            dynamic_ncols=True,
            ascii=True,
        ) as pbar:
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i : i + batch_size]
                Y_batch = Y_train[i : i + batch_size]

                optimizer.zero_grad()

                # Sample latent noise x0.
                x0 = noise_dist(X_batch.shape[0], X_batch.shape[1]).to(device)

                # Sample intermediate points and target conditional flow.
                t, xt, ut = fm.sample_location_and_conditional_flow(x0, X_batch)

                # Predict the flow field with the neural network.
                vt = model(xt, context=Y_batch, flow_time=t[:, None])

                # Mean squared error loss in velocity space.
                loss = torch.mean((vt - ut) ** 2)
                train_loss += loss.item()

                # Backpropagation step.
                loss.backward()
                optimizer.step()

                pbar.update(1)
                pbar.set_postfix({"Batch Loss": loss.item()})

        train_loss /= total_train_batches
        train_history.append(train_loss)
        writer.add_scalar("loss", train_loss, epoch)

        # Update scheduler.
        if scheduler is not None:
            if train_kwargs["scheduler"] == "ReduceLROnPlateau":
                scheduler.step(train_loss)
            else:
                scheduler.step()

        print(f"Epoch: {epoch} Loss: {train_loss:.6f}")

        # Evaluate on the test set.
        test_loss = compute_test_loss(
            model=model,
            fm=fm,
            X_test=X_test,
            Y_test=Y_test,
            test_batch_size=test_batch_size,
            noise_dist=noise_dist,
            device=device,
        )
        test_history.append(test_loss)
        writer.add_scalar("test_loss", test_loss, epoch)
        print(f"Test Loss: {test_loss:.6f}")

        # Save losses to a CSV file for later inspection.
        csv_file_path = os.path.join(save_dir, "losses.csv")
        mode = "a" if os.path.exists(csv_file_path) else "w"

        with open(csv_file_path, mode) as f:
            if mode == "w":
                f.write("epoch,train_loss,test_loss\n")
            f.write(f"{epoch},{train_loss},{test_loss}\n")

        # Periodically sample and validate.
        if epoch % train_kwargs["eval_freq"] == 0:
            print("Starting sampling")

            samples = sample_from_model(
                model=model,
                X_test=X_test,
                Y_test=Y_test,
                test_batch_size=test_batch_size,
                noise_dist=noise_dist,
                device=device,
                base_kwargs=base_kwargs,
                context_dim=context_dim,
            )

            # Bring generated samples back to physical space if needed.
            if data_kwargs["standardize"]:
                samples = test_dataset.scaler_x.inverse_transform(samples)

            # Round count-like generated features.
            samples = round_integer_like_features(samples)

            # Clip generated features to the physical range seen in the test data.
            samples = clip_samples_to_test_range(samples, X_test_cpu)

            # Save outputs used for later analysis.
            np.save(os.path.join(save_dir, "samples.npy"), samples)
            np.save(os.path.join(save_dir, "X_test_cpu.npy"), X_test_cpu)
            np.save(os.path.join(save_dir, "Y_test_cpu.npy"), Y_test_cpu)

            print("Starting evaluation")
            validate(samples, X_test_cpu, Y_test_cpu, save_dir, epoch, writer)

        # Periodically save model checkpoints.
        if epoch % train_kwargs["save_freq"] == 0:
            save_cfm_model(
                model=model,
                epoch=epoch,
                lr=print_lr,
                name=train_kwargs["log_name"],
                input_dim=input_dim,
                context_dim=context_dim,
                base_kwargs=base_kwargs,
                path=save_dir,
            )
            print("Saved model")


if __name__ == "__main__":
    # Read the YAML config file passed as command-line argument.
    # Example:
    #   python train_cfm.py CRT.yaml
    args = sys.argv

    if len(args) > 1:
        config_path = "../configs/" + args[1]
    else:
        config_path = "../configs/dummy_train_config.yaml"

    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)

    input_dim = config["input_dim"]
    context_dim = config["context_dim"]
    gpu = config["gpu"]
    train_kwargs = config["train_kwargs"]
    data_kwargs = config["data_kwargs"]
    base_kwargs = config["base_kwargs"]

    train(
        input_dim=input_dim,
        context_dim=context_dim,
        gpu=gpu,
        train_kwargs=train_kwargs,
        data_kwargs=data_kwargs,
        base_kwargs=base_kwargs,
    )