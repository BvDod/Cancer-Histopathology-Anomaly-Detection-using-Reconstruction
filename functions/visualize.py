import random
from PIL import Image
import torchvision.transforms as transforms
import torchvision

def plot_grid_samples_tensor(tensor, grid_size=[8,8]):
    """ Plots a grid of random samples from a tensor with grid size = grid size"""
    grid = torchvision.utils.make_grid(tensor, nrow=grid_size[0])
    return grid

