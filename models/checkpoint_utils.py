def get_prefix(task: str = "imagenet", model_name: str = "alexnet"):
    if task == "imagenet":
        return "module."

    # elif task == "lamem" or task == "lamem_shuffle":
    elif "lamem" in task:
        return "model.model."

    else:
        raise NotImplementedError

def remove_prefix(state_dict, prefix):
    """
    Remove a prefix from the state_dict keys.

    Args:
    state_dict (dict): State dictionary from which the prefix will be removed.
    prefix (str): Prefix to be removed.

    Returns:
    dict: State dictionary with prefix removed from keys.
    """
    return {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
        }

def match_and_load_weights(checkpoint_state_dict, model, prefix="module."):
    """
    Match weights from checkpoint_state_dict with model's state_dict and load them into the model.

    Args:
    checkpoint_state_dict (dict): State dictionary from checkpoint.
    model (torch.nn.Module): The model instance.
    prefix (str): Prefix to be removed from checkpoint keys.

    Returns:
    None
    """
    # Remove the prefix from checkpoint state dict keys
    cleaned_checkpoint_state_dict = remove_prefix(checkpoint_state_dict, prefix)

    model_state_dict = model.state_dict()
    matched_weights = {}

    # Iterate over the cleaned checkpoint state dict
    for ckpt_key, ckpt_weight in cleaned_checkpoint_state_dict.items():
        if ckpt_key in model_state_dict:
            # If the layer name matches, add to the matched_weights dict
            matched_weights[ckpt_key] = ckpt_weight
        else:
            print(
                f"Layer {ckpt_key} from checkpoint not found in the model state dict."
            )

    return matched_weights