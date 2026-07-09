from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from vit_model import VisionTransformer

def predict_my_image(image_path):
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    model = VisionTransformer(num_classes=10)
    model.load_state_dict(torch.load('vit_cifar10_weights.pth', map_location='cpu'))
    model.eval()

    my_image = Image.open(image_path).convert('RGB')

    custom_transform = transforms.Compose([
        transforms.Resize(32),        
        transforms.CenterCrop(32),    
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    input_tensor = custom_transform(my_image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted_idx = torch.max(outputs, 1)
        predicted_label = classes[predicted_idx[0].item()]
        
    print(f" Model predicts: [{predicted_label.upper()}]")

    plt.figure(figsize=(5, 5))
    plt.imshow(my_image)
    plt.title(f"ViT Prediction: {predicted_label.upper()}", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.show()


predict_my_image("test_images/sailboat(1).jpg")