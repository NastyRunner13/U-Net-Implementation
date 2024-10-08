# U-Net Implementation for Image Segmentation

This repository contains a PyTorch implementation of the U-Net architecture for image segmentation. The project is structured to train and evaluate a U-Net model on the Carvana Image Masking Challenge dataset.

## Project Structure

- `model.py`: Contains the U-Net model architecture implementation.
- `train.py`: Script for training the U-Net model.
- `utils.py`: Utility functions for data loading, checkpointing, and evaluation.
- `config.py`: Configuration file with hyperparameters and directory paths.
- `dataset.py`: Custom dataset class for loading and preprocessing images and masks.

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- albumentations
- tqdm
- Pillow
- NumPy

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/your-username/unet-implementation.git
   cd unet-implementation
   ```

2. Install the required packages:
   ```
   pip install torch torchvision albumentations tqdm Pillow numpy
   ```

3. Download the Carvana Image Masking Challenge dataset from Kaggle:
   https://www.kaggle.com/c/carvana-image-masking-challenge

   After downloading, organize the data into the following directory structure:
   ```
   data/
   ├── train_images/
   ├── train_masks/
   ├── val_images/
   └── val_masks/
   ```

## Usage

1. Adjust the hyperparameters and paths in `config.py` if necessary.

2. Train the model:
   ```
   python train.py
   ```

3. The trained model will be saved as `my_checkpoint.pth.tar`.

4. To use the trained model for inference, load the checkpoint in your script and use the `UNET` class from `model.py`.

## Model Architecture

The implemented U-Net architecture consists of:
- An encoder path with four double convolution blocks
- A bottleneck layer
- A decoder path with four up-convolution and double convolution blocks
- Skip connections between encoder and decoder blocks

The model uses batch normalization and ReLU activation functions. Optional dropout layers can be added for regularization.

## Customization

- Modify the `UNET` class in `model.py` to experiment with different architectures.
- Adjust training parameters in `config.py` to optimize performance.
- Implement additional data augmentation techniques in `config.py` using the albumentations library.

## Acknowledgements

This implementation is based on the original U-Net paper:
Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. In Medical Image Computing and Computer-Assisted Intervention (MICCAI), Springer, LNCS, Vol.9351: 234--241, 2015.

The dataset used in this project is from the Carvana Image Masking Challenge on Kaggle.
