import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json
import logging
from functools import wraps
import pickle

from data import get_dataset
from models import get_model, get_checkpoint_path, get_layer_name
from feature_extractor import extract_features, concat_all_features
from metrics import CKA

def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup a logger"""
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def log_exceptions(func):
    """Decorator to log exceptions"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            self.logger.exception(f"Exception occurred in {func.__name__}: {str(e)}")
            raise
    return wrapper

class FeatureAnalysisPipeline:
    def __init__(self, data_root, input_size=256, batch_size=32, num_workers=4, device=None, log_file='pipeline.log'):
        self.data_root = data_root
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.logger = setup_logger('FeatureAnalysisPipeline', log_file)
        self.logger.info(f"Initializing FeatureAnalysisPipeline with data_root: {data_root}, device: {self.device}")
        
        try:
            self.dataset = get_dataset(self.data_root, input_size=self.input_size)
            self.logger.info(f"Dataset loaded successfully. Size: {len(self.dataset)}")
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {str(e)}")
            raise

    @log_exceptions
    def _get_model(self, model_name, task, checkpoint_path=None, layer_names=None):
        self.logger.info(f"Loading model: {model_name}, task: {task}")
        if checkpoint_path is None:
            checkpoint_path = get_checkpoint_path(model_name, task)
        if layer_names is None:
            layer_names = get_layer_name(model_name)
    
        model = get_model(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            layer_name=layer_names,
            task=task,
            device=self.device,
        )
        model.eval()
        self.logger.info(f"Model {model_name} loaded successfully")
        return model

    @log_exceptions
    def extract_features_single_model(self, model_name, task, layer_names=None):
        self.logger.info(f"Extracting features for model: {model_name}, task: {task}")
        model = self._get_model(model_name, task, layer_names=layer_names)
        features_dict = extract_features(self.dataset, {"model": model}, layer_names, self.device, 
                                        batch_size=self.batch_size, num_workers=self.num_workers)
        self.logger.info(f"Features extracted successfully for model: {model_name}")
        return features_dict

    @log_exceptions
    def extract_features_two_models(self, model_name_1, model_name_2, task, layer_names=None):
        self.logger.info(f"Extracting features for models: {model_name_1} and {model_name_2}, task: {task}")
        model1 = self._get_model(model_name_1, task, layer_names=layer_names)
        model2 = self._get_model(model_name_2, task, layer_names=layer_names)
        
        models = {"model1": model1, "model2": model2}
        features_dict = extract_features(self.dataset, models, layer_names, self.device, 
                                         batch_size=self.batch_size, num_workers=self.num_workers)
        self.logger.info(f"Features extracted successfully for models: {model_name_1} and {model_name_2}")
        return concat_all_features(features_dict)

    @log_exceptions
    def calculate_cka(self, features1, features2):
        self.logger.info("Calculating CKA")
        return CKA(features1, features2)

    @log_exceptions
    def run_single_model_single_layer(self, model_name, task, layer_name):
        self.logger.info(f"Running single model single layer analysis: {model_name}, {task}, {layer_name}")
        features = self.extract_features_single_model(model_name, task, [layer_name])
        return {layer_name: features["model"][layer_name]}

    @log_exceptions
    def run_single_model_multiple_layers(self, model_name, task, layer_names=None):
        self.logger.info(f"Running single model multiple layers analysis: {model_name}, {task}")
        return self.extract_features_single_model(model_name, task, layer_names)["model"]

    @log_exceptions
    def run_two_models_single_layer(self, model_name_1, model_name_2, task, layer_name):
        self.logger.info(f"Running two models single layer analysis: {model_name_1}, {model_name_2}, {task}, {layer_name}")
        features = self.extract_features_two_models(model_name_1, model_name_2, task, [layer_name])
        cka_value = self.calculate_cka(features["model1"][layer_name], features["model2"][layer_name])
        return {layer_name: {"cka": cka_value, "shape1": features["model1"][layer_name].shape, "shape2": features["model2"][layer_name].shape}}

    @log_exceptions
    def run_two_models_multiple_layers(self, model_name_1, model_name_2, task, layer_names=None):
        self.logger.info(f"Running two models multiple layers analysis: {model_name_1}, {model_name_2}, {task}")
        features = self.extract_features_two_models(model_name_1, model_name_2, task, layer_names)
        results = {}
        for layer in features["model1"].keys():
            cka_value = self.calculate_cka(features["model1"][layer], features["model2"][layer])
            results[layer] = {"cka": cka_value, "shape1": features["model1"][layer].shape, "shape2": features["model2"][layer].shape}
        return results

    @log_exceptions
    def save_results(self, results, output_dir, filename):
        self.logger.info(f"Saving results to {output_dir}/{filename}")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, filename)
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        self.logger.info(f"Results saved successfully to {output_file}")

    @log_exceptions
    def visualize_results(self, results):
        self.logger.info("Visualizing results")
        # Implement visualization logic here
        # This could include bar charts, heatmaps, etc.
        pass