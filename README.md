# Mini-ViT-From-Scratch: CIFAR-10 Vision Transformer 

A complete, lightweight Vision Transformer (ViT) implemented from scratch using PyTorch. This project demonstrates the core mechanics of Vision Transformers (Patch Embeddings, Positional Encoding, Self-Attention, and MLP Heads) without relying on pre-built transformer libraries.

It is trained on the **CIFAR-10** dataset using Google Colab's T4 GPU and includes a local inference script to test the model on your own custom images.

##  Features
* **Built from Scratch:** Pure PyTorch implementation of the ViT architecture.
* **Cloud Training:** Optimized Jupyter Notebook for fast training on Google Colab GPUs.
* **Local Inference:** Ready-to-use Python script for testing the model on local images using OpenCV/PIL and Matplotlib.
* **Pre-trained Weights Included:** The repository contains the `.pth` weights file, so you can run inference immediately without retraining.

##  Project Structure
```text
├── vit_model.py               # The core Vision Transformer architecture classes
├── vit_training.ipynb         # Google Colab notebook for data loading and training
├── inference.py               # Local script to test the model on custom images
├── vit_cifar10_weights.pth    # Saved model weights (trained on CIFAR-10)
├── test_images/               # Folder containing sample images for testing
├── README.md                  # Project documentation
└── .gitignore                 # Ignored files and folders