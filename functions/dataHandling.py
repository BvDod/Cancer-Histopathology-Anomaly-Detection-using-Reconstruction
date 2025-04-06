""" Contains all functions regarding file/dataset handling """

import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import Subset

def get_dataset(dataset_name: str, print_stats = False, disable_augmentation = False):
    """ Retrieves dataset with name dataset name from disk and returns it"""

    if dataset_name == "CRC":
        if not disable_augmentation:
            img_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.2, hue=0.1),
                transforms.RandomAffine(3, scale=(0.95, 1.05)),
                transforms.ToTensor(),
            ])
            
        elif disable_augmentation:
            img_transforms = transforms.Compose([
                transforms.ToTensor(),]
            )

        transforms_val = transforms.Compose([
            transforms.ToTensor(),])
        
    else:   
        img_transforms = transforms.Compose([                                     
            transforms.ToTensor(),]
        )
    
    if dataset_name == "CRC":
        train = torchvision.datasets.ImageFolder("./datasets/CRC/val/", transform=img_transforms)
        test = torchvision.datasets.ImageFolder("./datasets/CRC/test/", transform=transforms_val)

        # Calculate 1/10th size
        subset_size = len(test) // 2

        import random
        # Randomly select indices
        indices = random.sample(range(len(test)), subset_size)

        # Create subset
        test = Subset(test, indices)

        input_shape = (224, 224)
        channels = 3
        var = 0.225 # Precomputed

    if print_stats:
        print(f"Training dataset shape: {train[0][0].shape}, samples = {len(train)}")
        print(f"Test dataset shape: {test[0][0].shape}, samples = {len(test)}")
        print("\n")

    return train, test, input_shape, channels, var



def get_variance(dataset):
    """ Returns the variance of the dataset , pretty slow because of no batching"""
    var = 0
    for image in dataset:
        var += image[0].var()
    return var / len(dataset)