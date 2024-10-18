import torch
from transformers import ViTForImageClassification

def load_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ViTForImageClassification.from_pretrained('vit_retinal_classification').to(device)
    return model, device
