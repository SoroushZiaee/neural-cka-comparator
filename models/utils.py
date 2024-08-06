from typing import List

def get_checkpoint_path(model_name: str, task: str, model_id: int = 1):
    path_dict = {
        "resnet101": {
            "imagenet": "/scratch/soroush1/memorability/weights/clf/resnet101-1/checkpoint_epoch_90_0.78.pth",
            "lamem": "/home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/weights/LaMem/resnet101/epoch=89-val_loss=0.01-training_loss=0.01.ckpt",
            "lamem_shuffle": "/home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/weights/LaMem_random/resnet101/epoch=89-val_loss=0.02-training_loss=0.02.ckpt",
            "lamem_pretrain_freeze": "/home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/weights/LaMem_pretrain_freeze/resnet101/epoch=89-val_loss=0.01-training_loss=0.01.ckpt",
            "lamem_shuffle_pretrain_freeze": "/home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/weights/LaMem_shuffle_pretrain_freeze/resnet101/epoch=89-val_loss=0.02-training_loss=0.02.ckpt",
            "lamem_pretrain_no_freeze": "/home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/weights/LaMem_pretrain_no_freeze/resnet101/epoch=89-val_loss=0.01-training_loss=0.01.ckpt",
            "lamem_random_pretrain_no_freeze": "/home/soroush1/projects/def-kohitij/soroush1/pretrain-imagenet/weights/LaMem_random_pretrain_no_freeze/resnet101/epoch=89-val_loss=0.02-training_loss=0.02.ckpt",
        },
    }

    return path_dict[model_name][task]

def get_layer_name(model_name: str) -> List[str]:
    layer_dict = {
        "resnet101": ['maxpool', 'layer1.1.add', 'layer2.0.add', 'layer2.3.add', 'layer3.1.add', 'layer3.4.add', 'layer3.7.add', 'layer3.10.add', 'layer3.13.add', 'layer3.16.add'],
    }

    return layer_dict[model_name]