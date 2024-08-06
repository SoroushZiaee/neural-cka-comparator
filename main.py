import torch
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import os

from data import get_dataset, visualize_dataset_samples
from models import get_model, get_checkpoint_path, get_layer_name
from feature_extractor import extract_features, concat_all_features
from metrics import CKA

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)

    # Load dataset
    ds = get_dataset(args.data_root, input_size=args.input_size)
    print(f"Dataset size: {len(ds)}")

    # Visualize dataset samples (optional)
    if args.visualize:
        visualize_dataset_samples(ds, num_samples=args.num_vis_samples)

    # Load models
    checkpoint_path = get_checkpoint_path(args.model_name, args.task, model_id=args.model_id)
    layer_names = get_layer_name(args.model_name)

    if args.single_layer:
        layer_names = [args.single_layer]
    elif args.layers:
        layer_names = args.layers

    model_type = {
        "weights": None,
        "non_weights": None
    }

    model_type["weights"] = get_model(
        model_name=args.model_name,
        checkpoint_path=checkpoint_path,
        layer_name=layer_names,
        task=args.task,
        device=device,
        use_blurpool=not args.no_blurpool,
    )
    model_type["weights"].eval()

    model_type["non_weights"] = get_model(
        model_name=args.model_name,
        checkpoint_path=None,
        layer_name=layer_names,
        task=args.task,
        device=device,
        use_blurpool=not args.no_blurpool,
    )
    model_type["non_weights"].eval()

    # Extract features
    features_dict = extract_features(ds, model_type, layer_names, device, 
                                     batch_size=args.batch_size, num_workers=args.num_workers)

    # Concatenate features
    features = concat_all_features(features_dict)

    # Calculate CKA
    results = {}
    for layer in layer_names:
        cka_value = CKA(features[layer]["weights"], features[layer]["non_weights"])
        results[layer] = {
            "cka": cka_value,
            "weights_shape": features[layer]["weights"].shape,
            "non_weights_shape": features[layer]["non_weights"].shape
        }
        print(f"CKA value for {layer}: {cka_value}")
        print(f"Shape of weights features: {features[layer]['weights'].shape}")
        print(f"Shape of non-weights features: {features[layer]['non_weights'].shape}")

    # Save results
    if args.save_results:
        import json
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, f"{args.model_name}_{args.task}_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural CKA Comparator")
    parser.add_argument("--data_root", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--model_name", type=str, default="resnet101", help="Model name")
    parser.add_argument("--task", type=str, default="imagenet", help="Task name")
    parser.add_argument("--model_id", type=int, default=1, help="Model ID for checkpoint")
    parser.add_argument("--input_size", type=int, default=256, help="Input image size")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--visualize", action="store_true", help="Visualize dataset samples")
    parser.add_argument("--num_vis_samples", type=int, default=20, help="Number of samples to visualize")
    parser.add_argument("--single_layer", type=str, help="Process only a single layer")
    parser.add_argument("--layers", nargs='+', help="List of layers to process")
    parser.add_argument("--no_blurpool", action="store_true", help="Disable BlurPool")
    parser.add_argument("--save_results", action="store_true", help="Save results to file")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")

    args = parser.parse_args()
    main(args)
    
# python main.py --data_root /path/to/your/data --model_name resnet101 --task imagenet --model_id 1 --input_size 256 --batch_size 32 --num_workers 4 --seed 42 --visualize --num_vis_samples 10 --layers layer1.1.add layer2.3.add layer3.4.add --save_results --output_dir ./results