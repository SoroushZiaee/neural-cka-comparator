# Neural CKA Comparator

This project focuses on the analysis and comparison of neural networks using Centered Kernel Alignment (CKA), a method for measuring similarity between different neural network representations. It provides tools to extract features from various models, compute CKA, and analyze the results.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Scripts and Tools](#scripts-and-tools)
- [Contributing](#contributing)
- [License](#license)

## Installation

To set up the environment, ensure you have Python installed. Follow these steps to install the necessary dependencies:

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/neural-cka-comparator.git
   cd neural-cka-comparator
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Extract Features**: Use the scripts in `bash/feature_extractor/` to extract features from various neural networks. For example, to extract features using VGG19, run:

   ```bash
   bash bash/feature_extractor/extract_features_vgg19.sh
   ```

2. **Calculate CKA**: Compute the CKA similarity between layers or models using the scripts provided. For example, to calculate CKA for all layers, run:

   ```bash
   bash bash/feature_extractor/calculate_cka.sh
   ```

3. **Run Training**: Use the training scripts in `bash/training_script/` for classification or regression tasks. For example, to train using VGG16 for classification, run:

   ```bash
   bash bash/training_script/classification/training_vgg16.sh
   ```

## Project Structure

- `bash/`: Contains shell scripts for dataset preparation, feature extraction, and model training.
- `configs/`: Configuration files for running experiments.
- `data/`: Scripts for data loading, transformation, and visualization.
- `feature_extractor/`: Core implementation of feature extraction and CKA computation.
- `metrics/`: CKA calculation tools and utilities.
- `models/`: Model definitions and utilities for exporting and managing models.
- `pipe/`: Pipelines for feature analysis.

## Scripts and Tools

- **Feature Extraction**: Extract features from pre-trained models using scripts in `bash/feature_extractor/`.
- **CKA Calculation**: Compute CKA values using scripts like `calculate_cka.sh`.
- **Training Scripts**: Train models for different tasks using the scripts in `bash/training_script/`.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
