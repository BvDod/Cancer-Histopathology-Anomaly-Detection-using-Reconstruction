# %%
from functions.dataHandling import get_dataset
from functions.visualize import plot_grid_samples_tensor

import torch
from torch.utils.data import DataLoader

from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from PIL import Image as im 

from torchvision.models import resnet18

import numpy as np

def build_model_input(x_train, recon_model, input_types):
    input_tensor = torch.Tensor()

    if "reconstruction" in input_types or "error_map" in input_types:
        recon_model.eval()
        with torch.no_grad:
            recon, vq_loss = recon_model(x_train)

    if "image" in input_types:
        input_tensor = torch.cat((input_tensor, x_train))
    
    if "reconstruction" in input_types:
        input_tensor = torch.cat((input_tensor, recon))

    if "error_map" in input_types:
        error_map = torch.nn.functional.mse_loss(x_train, recon, reduction=None)
        error_map = torch.mean(error_map, dim=0)
        input_tensor = torch.cat((input_tensor, error_map))

    return input_tensor



def train_cnn(settings):

    # Tensorboard for logging
    writer = SummaryWriter()

    # Print settings and info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(str(settings))
    print(f"Device: {device}" + "\n")

    # Loading dataset
    train, test, input_shape, channels, train_var = get_dataset(settings["dataset"], print_stats=True)
    channels = len(settings["input_types"]) * 3 - (int("error_map" in settings["input_types"]) * 2)

    dataloader_train = DataLoader(train, batch_size=settings["batch_size"], shuffle=True, drop_last=True, pin_memory=False, num_workers=8)
    dataloader_test = DataLoader(test, batch_size=128, pin_memory=False, num_workers=8, shuffle=True)

    # Setting up model
    model_settings = settings["model_settings"]
    model_settings["num_channels"] = channels
    model_settings["input_shape"] = input_shape

    model = resnet18(num_classes=9).to(device)

    # Replace the first convolution layer to support custom amount of in channels
    model.conv1 = torch.nn.Conv2d(
    in_channels=channels,
    out_channels=64,       
    kernel_size=7,
    stride=2,
    padding=3,
    bias=False
    )

    if "error_map" in model_settings["input_types"] or "reconstruction" in model_settings["input_types"]:
        model_settings = None
        model_settings = settings["model_settings"]
        model_settings["num_channels"] = channels
        model_settings["input_shape"] = input_shape
        recon_model = VQVAE(model_settings).to(device)
        recon_model.load_state_dict(torch.load("reconstruction.pt", weights_only=True))
        recon_model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss()
    # scaler = torch.amp.GradScaler("cuda" ,enabled=True)


    # Training loop
    train_losses, test_losses = [], []
    best_test_loss = float("inf")
    for epoch in range(settings["max_epochs"]):
        train_losses_epoch = []
        print(f"Epoch: {epoch}/{settings['max_epochs']}")
        
        # Training
        model.train()
        correct = 0
        for batch_i, (x_train, y_train) in enumerate(dataloader_train):
            
            x_train = build_model_input(x_train, recon_model, model_settings["input_types"])
            
            x_train = x_train.to(device)
            #with torch.autocast(device_type=device, dtype=torch.float16, enabled=True):
            pred = model(x_train)
            mse = criterion(pred, y_train)

            _, predicted = torch.max(pred.detach(), 1)
            correct += (predicted == y_train).sum().item()
            loss = mse
            loss.backward()
            optimizer.step()
            #scaler.scale(loss).backward()
            #scaler.step(optimizer)
            #scaler.update()
            train_losses_epoch.append(loss.item())
            optimizer.zero_grad()
            print(train_losses_epoch[-1])
            break
            

           
        print(f"Train loss: {sum(train_losses_epoch) / len(train_losses_epoch)}")
        print(f"Train accuracy: {correct/len(train)}")
        train_losses.append(sum(train_losses_epoch) / len(train_losses_epoch))
        writer.add_scalar("Loss/train", train_losses[-1], epoch)
        writer.add_scalar("ACC/train", correct/len(train), epoch)
        
        """
        #  Early stopping
        epoch_delta = settings["early_stopping_epochs"]
        if len(train_losses) > epoch_delta and max(train_losses[-epoch_delta:-1]) < train_losses[-1]:
            print("Early stopping")
            break"
        """
        
        # Evaluation
        model.eval()
        correct = 0
        with torch.no_grad():
            test_losses_epoch = []
            test_mse_epoch = []
            for x_test, y_test in dataloader_test:
                x_test = x_test.to(device)
                #with torch.autocast(device_type=device, dtype=torch.float16, enabled=True):
                pred = model(x_test)
                mse = criterion(pred, y_test)
                loss = mse
                test_losses_epoch.append(loss.item())
                test_mse_epoch.append(mse.item())

                _, predicted = torch.max(pred, 1)
                correct += (predicted == y_test).sum().item()
                break
            print(f"Val accuracy: {correct/len(test)}")
            
            print(f"Test loss: {sum(test_losses_epoch) / len(test_losses_epoch)}")
            test_losses.append(sum(test_losses_epoch) / len(test_losses_epoch))

            writer.add_scalar("Loss/val", test_losses[-1], epoch)
            writer.add_scalar("ACC/val", correct/len(test), epoch)
        



if __name__ == "__main__":
    settings = {
        "dataset": "CRC",
        "save_model": True,

        "print_debug": False,
        "example_image_amount": 4,
        "save_reconstructions_first_epoch": True,
        "batch_size": 32,
        "learning_rate": 3e-4, # for x-ray
        "max_epochs": 100000,
        "early_stopping_epochs": 3,
        "input_types": ["image", "reconstruction", "error_map"],



        "model_settings" : {
            "encoder_architecture": "VIT",
            "decoder_architecture": "VIT",
            "num_hidden": 128,
            "num_residual_hidden": 128,
            "embedding_dim": 64,
            "num_embeddings": 512,
            "commitment_cost": 0.25,
            "transformer_layers": 5,
            "attention_heads": 4,
            "patch_size": 8,
        }
    }
    train_cnn(settings)