from torchvision import transforms
from .dataset import MuriDataset


def get_dataset(root: str, input_size: int = 256):

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    imagenet_transform = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return MuriDataset(root=root, transforms=imagenet_transform)