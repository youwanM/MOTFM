import argparse
import os
import sys
import time
import warnings
import datetime

# Suppress most warnings for cleaner logs (comment out if debugging is needed)
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from monai.data import DataLoader, Dataset
from tqdm import tqdm
from monai import transforms
import wandb

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath

from utils.general_utils import (
    load_config,
    load_and_prepare_data,
    create_dataloader,
    save_checkpoint,
    load_checkpoint,
)
from utils.utils_fm import build_model, validate_and_save_samples


def main():
    # Parse arguments and load config
    parser = argparse.ArgumentParser(description="Train the flow matching model.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/default.yaml",
        help="Path to the configuration file.",
    )
    args = parser.parse_args()
    config_path = args.config_path
    config = load_config(config_path)

    # Read core settings from config
    num_epochs = config["train_args"]["num_epochs"]
    num_val_samples = config["train_args"].get("num_val_samples", 5)
    batch_size = config["train_args"]["batch_size"]
    lr = config["train_args"]["lr"]
    print_every = config["train_args"].get("print_every", 1)
    val_freq = config["train_args"].get("val_freq", 5)
    root_ckpt_dir = config["train_args"]["checkpoint_dir"]

    # Decide which device to use
    device = (
        torch.device(config["train_args"]["device"])
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print("Using device:", device)

    # Model configuration flags
    mask_conditioning = config["general_args"]["mask_conditioning"]
    class_conditioning = config["general_args"]["class_conditioning"]

    # Build model
    model = build_model(config["model_args"], device=device)

    # Prepare data
    train_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.Orientationd(keys=["image"], axcodes="RAS"),
        transforms.ResizeWithPadOrCropd(keys=["image"], spatial_size=(182, 218, 182)),
        transforms.Resized(keys=["image"], spatial_size=(128, 128, 128)),
        transforms.NormalizeIntensityd(keys=["image"]),
    ]
    )

    csv_file = "data/train-IXI-T1-preproc.csv" 
    df = pd.read_csv(csv_file)
    data_list = [{"image": path} for path in df["filepaths"]]

    train_ds = Dataset(data=data_list, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True)

    #Wandb Logging
        # Define phonetic alphabet
    phonetic_alphabet = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel", "India", "Juliett", "Kilo", "Lima", "Mike", "November", "Oscar", "Papa", "Quebec", "Romeo", "Sierra", "Tango", "Uniform", "Victor", "Whiskey", "X-ray", "Yankee", "Zulu"]

    # Get current date and time
    now = datetime.datetime.now()

    # Determine the letters based on the current hour and day of the month
    hour_index = now.hour % len(phonetic_alphabet)
    day_index = now.day % len(phonetic_alphabet)
    additional_code = ""
    if now.day > 25:
        additional_code = f"-{phonetic_alphabet[(now.day - 26) % len(phonetic_alphabet)]}"
        run_name = f"{phonetic_alphabet[hour_index]} {phonetic_alphabet[day_index]} {additional_code}"
    else:
        run_name = f"{phonetic_alphabet[hour_index]} {phonetic_alphabet[day_index]}"
    wandb.init(project="MOTFM", config=config, name=run_name)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Load the latest checkpoint if available
    latest_ckpt_dir = os.path.join(root_ckpt_dir, "latest")
    start_epoch, loaded_config = load_checkpoint(
        model, optimizer, checkpoint_dir=latest_ckpt_dir, device=device, valid_only=False
    )

    # Define path object (scheduler included)
    path = AffineProbPath(scheduler=CondOTScheduler())

    solver_config = config["solver_args"]

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        # Use tqdm for the train loader to get a per-batch progress bar
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            im_batch = batch["image"].to(device)
            mask_batch = batch["mask"].to(device) if mask_conditioning else None
            classes_batch = batch["classe"].to(device).unsqueeze(1) if class_conditioning else None

            # Sample random initial noise, and random t
            x_0 = torch.randn_like(im_batch)
            t = torch.rand(im_batch.shape[0], device=device)

            # Sample the path from x_0 to x_batch
            sample_info = path.sample(t=t, x_0=x_0, x_1=im_batch)

            # Predict velocity and compute loss
            v_pred = model(
                x=sample_info.x_t,
                t=sample_info.t,
                masks=mask_batch,
                cond=classes_batch,
            )
            loss = F.mse_loss(v_pred, sample_info.dx_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Log the loss at each iteration
            wandb.log({"iteration_loss": loss.item()})

        # Logging the average loss at the end of the epoch
        avg_loss = epoch_loss / len(train_loader)
        wandb.log({"epoch": epoch + 1, "epoch_loss": avg_loss})

        # Validation & checkpoint saving
        if (epoch + 1) % val_freq == 0:
            epoch_ckpt_dir = os.path.join(root_ckpt_dir, f"epoch_{epoch+1}")
            save_checkpoint(model, optimizer, epoch + 1, config, epoch_ckpt_dir)
            save_checkpoint(model, optimizer, epoch + 1, config, latest_ckpt_dir)

            # Validation
            validate_and_save_samples(
                model=model,
                val_loader=val_loader,
                device=device,
                checkpoint_dir=epoch_ckpt_dir,
                epoch=epoch + 1,
                solver_config=solver_config,
                max_samples=num_val_samples,
                class_map=train_data["class_map"],
                mask_conditioning=mask_conditioning,
                class_conditioning=class_conditioning,
            )

    print("Training complete!")


if __name__ == "__main__":
    main()
