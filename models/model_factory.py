import torch

from typing import List, Optional
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from .blurpool import apply_blurpool
from .checkpoint_utils import match_and_load_weights, get_prefix

def get_model(
    model_name,
    checkpoint_path: str = None,
    layer_name: Optional[List[str]] = None,
    use_blurpool: bool = True,
    task: str = "imagenet",
    device="cpu",
):
    """
    Create a model from torchvision.models and load weights from checkpoint if provided.

    Args:
    model_name (str): Name of the model to be created.
    checkpoint_path (str): Path to the checkpoint file.
    layer_name (str): Name of the layer to extract features from.
    use_blurpool (bool): Whether to use BlurPoolConv2d for convolution layers with stride > 1.
    task (str): Whether to be imagenet, lamem, or combine


    Returns:
    torch.nn.Module: The model instance.
    """

    def correct_checkpoint(checkpoint):
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]

        return checkpoint

    if isinstance(layer_name, str):
        layer_name = [layer_name]

    
    model = getattr(models, model_name)(weights=None)
    
    if use_blurpool:
        apply_blurpool(model)

    # TODO: Missing key(s) in state_dict: "model.features.0.weight"
    # if task == "lamem":
    #     model = apply_regression(model)
    
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint = correct_checkpoint(checkpoint)
        # print(f"{checkpoint.keys() = }")
        prefix = get_prefix(task, model_name)
        # print(f"{checkpoint.keys() = }")
        matched_weights = match_and_load_weights(checkpoint, model, prefix=prefix)
        model.load_state_dict(matched_weights)
        # 1. Check if all keys in matched_weights are in model's state_dict
        model_state_dict = model.state_dict()

        # 3. Check if any weights have changed (compare a few values)
        print("Checking if weights have changed:")
        for key in list(matched_weights.keys())[:5]:  # Check first 5 layers
            if key in model_state_dict:
                checkpoint_vals = matched_weights[key].flatten()[:5]  # First 5 values
                model_vals = model_state_dict[key].flatten()[:5]
                print(f"Layer {key}:")
                print(f"  Checkpoint values: {checkpoint_vals}")
                print(f"  Model values:      {model_vals}")
                if not torch.allclose(checkpoint_vals, model_vals):
                    print(f"  Warning: Values don't match in layer {key}")
    
        print("Weight loading verification complete.")

    else:
        print(f"loadding pytorch pre-trained model")
        model = getattr(models, model_name)(weights="DEFAULT")

    if layer_name:
        model = create_feature_extractor(model, layer_name)

    model.to(device)

    return model
