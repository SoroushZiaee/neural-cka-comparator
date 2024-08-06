import argparse
import yaml
from pipe.feature_analysis_pipeline import FeatureAnalysisPipeline

def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def test_single_model_single_layer(config):
    pipeline = FeatureAnalysisPipeline(config['data_root'], input_size=config['input_size'], batch_size=config['batch_size'])
    results = pipeline.run_single_model_single_layer(config['model_name'], config['task'], config['layer_name'])
    
    if config['save_results']:
        pipeline.save_results(results, config['output_dir'], "results.pkl")
    print(f"Results saved to {config['output_dir']}/results.json")

def test_single_model_multiple_layers(config):
    pipeline = FeatureAnalysisPipeline(config['data_root'], input_size=config['input_size'], batch_size=config['batch_size'])
    results = pipeline.run_single_model_multiple_layers(config['model_name'], config['task'], config['layer_names'])
    if config['save_results']:
        pipeline.save_results(results, config['output_dir'], "results.pkl")
    print(f"Results saved to {config['output_dir']}/results.json")

def test_two_models_single_layer(config):
    pipeline = FeatureAnalysisPipeline(config['data_root'], input_size=config['input_size'], batch_size=config['batch_size'])
    results = pipeline.run_two_models_single_layer(config['model_name_1'], config['model_name_2'], config['task'], config['layer_name'])
    if config['save_results']:
        pipeline.save_results(results, config['output_dir'], "results.pkl")
    print(f"Results saved to {config['output_dir']}/results.json")

def test_two_models_multiple_layers(config):
    pipeline = FeatureAnalysisPipeline(config['data_root'], input_size=config['input_size'], batch_size=config['batch_size'])
    results = pipeline.run_two_models_multiple_layers(config['model_name_1'], config['model_name_2'], config['task'], config['layer_names'])
    if config['save_results']:
        pipeline.save_results(results, config['output_dir'], "results.pkl")
    print(f"Results saved to {config['output_dir']}/results.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test FeatureAnalysisPipeline methods")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()

    config = load_config(args.config)

    if 'model_name_2' not in config:
        if isinstance(config['layer_name'], list) or 'layer_names' in config:
            test_single_model_multiple_layers(config)
        else:
            test_single_model_single_layer(config)
    else:
        if isinstance(config['layer_name'], list) or 'layer_names' in config:
            test_two_models_multiple_layers(config)
        else:
            test_two_models_single_layer(config)