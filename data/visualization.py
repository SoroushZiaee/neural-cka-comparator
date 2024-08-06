import random
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def visualize_dataset_samples(dataset, num_samples=20):
    # Get a list of random indices
    indices = random.sample(range(len(dataset)), num_samples)
    
    # Get the images and labels
    images = []
    labels = []
    for idx in indices:
        image, _ = dataset[idx]
        images.append(image)
    
    # Convert list of images to a tensor
    images = torch.stack(images)
    
    # Create a grid of images
    grid = make_grid(images, nrow=5, normalize=True)
    
    # Convert to numpy for displaying
    plt.figure(figsize=(15, 15))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.title('Sample Images from Dataset')
    plt.show()