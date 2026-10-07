Mini-ViT-From-Scratch & Hybrid CNN-ViT: CIFAR-10 Vision Transformers
A complete, lightweight Vision Transformer (ViT) and a Hybrid CNN-ViT model implemented from scratch using PyTorch. This project demonstrates the core mechanics of Vision Transformers (Patch Embeddings, Positional Encoding, Self-Attention, and MLP Heads) and how they can be combined with a Convolutional Neural Network (CNN) backbone for enhanced feature extraction, without relying on pre-built transformer libraries.

Both models are designed to be trained on the CIFAR-10 dataset using Google Colab's T4 GPU and include a local inference script to test the architectures on your own custom images.
##  Features
* **Built from Scratch:** Pure PyTorch implementation of the ViT architecture.
* **Cloud Training:** Optimized Jupyter Notebook for fast training on Google Colab GPUs.
* **Local Inference:** Ready-to-use Python script for testing the model on local images using OpenCV/PIL and Matplotlib.
* **Pre-trained Weights Included:** The repository contains the `.pth` weights file, so you can run inference immediately without retraining.

##  Project Structure
```text
├── vit_model.py               # The core Vision Transformer architecture classes
├── cnn_vit_hybrid_model.py    # The hybrid CNN and ViT Model              
├── vit_training.ipynb         # Google Colab notebook for data loading and training
├── inference.py               # Local script to test the model on custom images
├── vit_cifar10_weights.pth    # Saved model weights (trained on CIFAR-10)
├── hybrid_cifar10_weights.pth # Saved hybrid model weights (trained on CIFAR-10)
├── test_images/               # Folder containing sample images for testing
├── README.md                  # Project documentation
└── .gitignore                 # Ignored files and folders