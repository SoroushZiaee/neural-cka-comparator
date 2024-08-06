import numpy as np

def concat_features(features_dict, layer, feature_type):
    """
    Concatenate features for all images in a single layer and feature type.
    
    :param features_dict: The dictionary containing all features
    :param layer: The layer to process (e.g., "layer3.16.add")
    :param feature_type: The feature type to process (e.g., "weights")
    :return: A numpy array containing concatenated features for all images
    """
    # Get the relevant sub-dictionary
    layer_features = features_dict[layer][feature_type]
    
    # Sort the image names to ensure consistent order
    image_names = sorted(layer_features.keys())
    
    # Concatenate the features
    concatenated_features = np.concatenate([layer_features[name] for name in image_names], axis=0)
    
    return concatenated_features


def concat_all_features(features_dict):
    """
    Concatenate features for all layers and feature types.
    
    :param features_dict: The dictionary containing all features
    :return: A dictionary with concatenated features for each layer and feature type
    """
    concatenated_features = {}
    
    for layer in features_dict:
        concatenated_features[layer] = {}
        for feature_type in features_dict[layer]:
            concatenated_features[layer][feature_type] = concat_features(features_dict, layer, feature_type)
    
    return concatenated_features