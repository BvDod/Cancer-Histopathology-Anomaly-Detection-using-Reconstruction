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
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC

import numpy as np

from models.vq_vae import VQVAE

def build_model_input(x_train, y_train, recon_model, input_types, writer=None):
    input_tensor = torch.Tensor().to('cuda')
    x_train = x_train.to('cuda')

    
    if "reconstruction" in input_types or "error_map" in input_types:
        recon_model.eval()
        
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                recon, vq_loss = recon_model(x_train)

    if "image" in input_types:
        input_tensor = torch.cat((input_tensor, x_train))
    
    if "reconstruction" in input_types:
        input_tensor = torch.cat((input_tensor, recon), dim=1)

        grid = plot_grid_samples_tensor(recon[:9])
        writer.add_image("reconstruction", grid, 0)
        
        grid = plot_grid_samples_tensor(x_train[:9])
        writer.add_image("input image", grid, 0)

    if "error_map" in input_types:
        error_map = torch.nn.functional.mse_loss(x_train, recon, reduction='none')
        error_map = torch.mean(error_map, dim=1)
        error_map = error_map.unsqueeze(1)
        error_map = error_map / (0.5/5)
        input_tensor = torch.cat((input_tensor, error_map), dim=1)

        grid = plot_grid_samples_tensor(error_map[:9])
        writer.add_image("error_map", grid, 0, dataformats="CHW")


    return input_tensor

import torch.nn as nn

# ----- Simple CNN -----
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

        self.features = nn.Sequential(
            conv_block(4, 32),      # -> (112x112)
            conv_block(32, 64),     # -> (56x56)
            conv_block(64, 128),    # -> (28x28)
            conv_block(128, 256),   # -> (14x14)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

    
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

    dataloader_train = DataLoader(train, batch_size = 16, shuffle=True, drop_last=True, num_workers=6)
    dataloader_test = DataLoader(test, batch_size= 16, pin_memory=False, shuffle=True, num_workers=6)

    # Setting up model
    model_settings = settings["model_settings"]
    model_settings["num_channels"] = channels
    model_settings["input_shape"] = input_shape

    """
    model = SimpleCNN().to(device)

    # Replace the first convolution layer to support custom amount of in channels
    model.conv1 = torch.nn.Conv2d(
    in_channels=channels,
    out_channels=64,       
    kernel_size=7,
    stride=2,
    padding=3,
    bias=False,
    device=device,
    )
    """

    
    if "error_map" in settings["input_types"] or "reconstruction" in settings["input_types"]:
        model_settings = settings["model_settings"]
        model_settings["num_channels"] = 3
        model_settings["input_shape"] = input_shape
        recon_model = VQVAE(model_settings).to(device)
        recon_model.load_state_dict(torch.load("reconstruction_latest.pt", weights_only=True))
        recon_model.to(device)
        recon_model.eval()
    else:
        recon_model = None
    
    #optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss(weight= torch.tensor([2/9, 7/9], device=device))
    scaler = torch.amp.GradScaler("cuda" ,enabled=True)

    accuracy = BinaryAccuracy()
    precision = BinaryPrecision()
    recall = BinaryRecall()
    f1 = BinaryF1Score()

    # Training loop
    train_losses, test_losses = [], []
    best_test_loss = float("inf")
    for epoch in range(settings["max_epochs"]):
        train_losses_epoch = []
        print(f"\n\nEpoch: {epoch}/{settings['max_epochs']}")
        
        # Training
        #model.train()
        correct = 0

        avg_error_no_cancer = 0
        no_cancer_n = 0
        avg_error_cancer = 0
        cancer_n = 0
        with torch.no_grad():
            for batch_i, (x_train, y_train) in enumerate(dataloader_train):
                y_train = (y_train > 6).long()
                x_train = build_model_input(x_train, y_train, recon_model, settings["input_types"], writer=writer)
                
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=True):
                    x_train = x_train.to(device)
                    print(x_train.mean(dim=[1,2,3]).shape)
                    x_train_cancer = x_train[y_train == 1].cpu()
                    x_train_no_cancer = x_train[y_train == 0].cpu()
                    cancer_n += len(x_train_cancer)
                    no_cancer_n += len(x_train_no_cancer)
                    avg_error_cancer += x_train_cancer.mean(dim=[1,2,3]).sum().item()
                    avg_error_no_cancer += x_train_no_cancer.mean(dim=[1,2,3]).sum().item()
                
            
        print(f"Avg error no cancer: {avg_error_no_cancer / no_cancer_n}")
        print(f"Avg error cancer: {avg_error_cancer / cancer_n}")

        exit()
            
           
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
            y_test_total = torch.Tensor()
            predicted_total = torch.Tensor()
            for batch_i, (x_test, y_test) in enumerate(dataloader_test):
                if batch_i == 100:
                    break
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=True):
                    y_test = (y_test > 6).long()
                    x_test = build_model_input(x_test, y_test, recon_model, settings["input_types"], writer=writer)
                    x_test = x_test.to(device)
                    y_test = y_test.to(device)
                    pred = model(x_test)
                    mse = criterion(pred, y_test)
                    loss = mse
                test_losses_epoch.append(loss.item())
                test_mse_epoch.append(mse.item())

                _, predicted = torch.max(pred, 1)
                correct += (predicted == y_test).sum().item()
                predicted_total = torch.cat((predicted_total, predicted.cpu()))
                y_test_total = torch.cat((y_test_total, y_test.cpu()))
            print(f"\nVal accuracy: {correct/len(test)}")
            
            print(f"Test loss: {sum(test_losses_epoch) / len(test_losses_epoch)}")
            test_losses.append(sum(test_losses_epoch) / len(test_losses_epoch))

            writer.add_scalar("Loss/val", test_losses[-1], epoch)
            writer.add_scalar("ACC/val", correct/len(test), epoch)
            


            print("Accuracy:", accuracy(predicted_total, y_test_total).item())
            print("Precision:", precision(predicted_total,  y_test_total).item())
            print("Recall:", recall(predicted_total,  y_test_total).item())
            print("F1 Score:", f1(predicted_total,  y_test_total).item())
        



if __name__ == "__main__":
    settings = {
        "dataset": "CRC",
        "save_model": True,

        "print_debug": False,
        "example_image_amount": 4,
        "save_reconstructions_first_epoch": True,
        "batch_size": 16,
        "learning_rate": 1e-4, # for x-ray
        "max_epochs": 150,
        "early_stopping_epochs": 3,
        "input_types": ["error_map"],



        "model_settings" : {
            "encoder_architecture": "VIT",
            "decoder_architecture": "CNN",
            "num_hidden": 128,
            "num_residual_hidden": 128,
            "embedding_dim": 64,
            "num_embeddings": 64,
            "commitment_cost": 0.25,
            "transformer_layers": 5,
            "attention_heads": 4,
            "patch_size": 4,
        }
    }
    train_cnn(settings)