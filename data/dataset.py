import os
from torch.utils.data import Dataset
import PIL.Image
from torchvision import transforms

class MuriDataset(Dataset):
    def __init__(self, root: str = "data/muri1320", transforms=None):

        self.root = root
        self.transforms = transforms
        img_path_list = os.listdir(root)
        img_path_list.sort()

        self.img_path_list = img_path_list

        self.image_name_template = "im{:04d}.png"

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root, self.image_name_template.format(idx))
        # print(f"{img_name = }")
        image = PIL.Image.open(img_name).convert("RGB")

        if self.transforms:
            image = self.transforms(image)

        return image, self.image_name_template.format(idx)

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