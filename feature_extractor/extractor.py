import torch
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import gc

def process_batch(batch, model_type, layer, device):
    """Process a batch of samples and extract features."""
    data, data_name = batch
    data = data.to(device)
    features = {}
    
    for key, value in model_type.items():
        with torch.no_grad():
            temp = value(data)
            temp = temp[layer]
            features[key] = temp.cpu().numpy()
    
    del temp
    torch.cuda.empty_cache()
    return data_name, features

def extract_features_for_layer(ds, model_type, layer, device, batch_size=32, num_workers=4):
    """Extract features for a single layer across all samples."""
    data_loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    
    features = defaultdict(dict)
    
    with tqdm(total=len(data_loader), desc=f"Extracting features for layer {layer}") as pbar:
        for batch in data_loader:
            batch_names, batch_features = process_batch(batch, model_type, layer, device)
            
            for key, feature_batch in batch_features.items():
                for name, feature in zip(batch_names, feature_batch):
                    features[key][name] = feature
            
            pbar.update(1)
    
    return features

def extract_features(ds, model_type, layer_names, device, batch_size=32, num_workers=4):
    """Main function to extract features for all specified layers."""
    features_dict = {"model": {}}
    
    
    for layer in tqdm(layer_names, desc="Processing layers"):
        features = extract_features_for_layer(ds, model_type, layer, device, batch_size, num_workers)
        features_dict["model"][layer] = features
        
        gc.collect()
        torch.cuda.empty_cache()
    
    return features_dict

def reshape_features(features):
    """Reshape the extracted features."""
    output = defaultdict(dict)

    for key, value in features.items():
        for img_name, feature in value.items():
            output[key][img_name] = feature.reshape(feature.shape[0], -1)
    
    return dict(output)