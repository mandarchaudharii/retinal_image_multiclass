import torch
import numpy as np
from PIL import Image
import torchvision.transforms.v2 as transforms

PRET_MEANS = [0.485, 0.456, 0.406]
PRET_STDS = [0.229, 0.224, 0.225]

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=PRET_MEANS, std=PRET_STDS)
])

def load_and_preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert('RGB')
    image_tensor = test_transform(image)
    return image_tensor

def integrated_gradients(model, img, label, steps=50):
    model.eval()
    label = label.unsqueeze(0) if label.ndim == 1 else label
    img = img.unsqueeze(0).requires_grad_(True)
    baseline = torch.zeros_like(img)
    scaled_images = [baseline + (float(i) / steps) * (img - baseline) for i in range(steps + 1)]
    
    grads = []
    for scaled_img in scaled_images:
        output = model(scaled_img).logits
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output, label.float())
        model.zero_grad()
        loss.backward()
        grads.append(img.grad.data.cpu().numpy())
    
    avg_grads = np.mean(grads, axis=0)
    integrated_grad = (img - baseline).detach().cpu().numpy() * avg_grads
    integrated_grad = np.maximum(integrated_grad.sum(axis=1), 0)
    integrated_grad -= integrated_grad.min()
    integrated_grad /= integrated_grad.max()
    
    return integrated_grad
