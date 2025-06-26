# MNIST CNN Example

This example demonstrates training a simple CNN model on MNIST using PyTorch and tracking the experiment with Tracely.

## Setup

1. First, ensure you have Python 3.7+ installed on your system.

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# OR
.\venv\Scripts\activate  # On Windows
```

3. Install the required packages:
```bash
pip install -r examples/mnist-cnn/requirements.txt
```
4. Install Tracely from the root directory:
```bash
pip install -e .
```

## Project Structure

- `mnist_training.py`: Main training script containing the CNN model and training loop
- `checkpoints/`: Directory where the best model will be saved (created automatically)
- `data/`: MNIST dataset will be downloaded here automatically

## Running the Example

1. Make sure you have Tracely properly configured. If not, follow the main Tracely documentation for setup instructions.

2. Run the training script:
```bash
python mnist_training.py
```

The script will:
- Download the MNIST dataset automatically on first run
- Train a CNN model for 5 epochs
- Log metrics to Tracely including:
  - Batch loss and accuracy
  - Validation loss and accuracy
  - Total training time
- Save the best model based on validation accuracy

## Monitoring Training

You can monitor the training progress through:
1. Console output showing validation metrics after each epoch
2. Tracely's web interface where you can view:
   - Real-time training metrics
   - Learning curves
   - Model artifacts
   - Run configuration
   - System metrics

## Troubleshooting

If you encounter CUDA-related errors:
1. Ensure you have CUDA installed if using an NVIDIA GPU
2. The script will automatically fall back to CPU if CUDA is not available
3. You can force CPU training by modifying the device in the config dictionary


