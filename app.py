import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
import streamlit as st
from PIL import Image
from transformers import ViTForImageClassification
import torchvision.transforms.v2 as transforms

# Define the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the labels
labels_list = ['ID', 'Disease_Risk', 'DR', 'ARMD', 'MH', 'DN', 'MYA', 'BRVO', 'TSLN',
               'ERM', 'LS', 'MS', 'CSR', 'ODC', 'CRVO', 'TV', 'AH', 'ODP', 'ODE', 'ST',
               'AION', 'PT', 'RT', 'RS', 'CRS', 'EDN', 'RPEC', 'MHL', 'RP', 'CWS',
               'CB', 'ODPM', 'PRH', 'MNF', 'HR', 'CRAO', 'TD', 'CME', 'PTCR', 'CF',
               'VH', 'MCA', 'VS', 'BRAO', 'PLQ', 'HPED', 'CL']

# Define image transforms
PRET_MEANS = [0.485, 0.456, 0.406]
PRET_STDS = [0.229, 0.224, 0.225]

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=PRET_MEANS, std=PRET_STDS)
])

# Load the model
vit_model = ViTForImageClassification.from_pretrained('vit_retinal_classification')
vit_model.to(device)

def integrated_gradients(model, img, label, steps=50):
    model.eval()
    label = label.unsqueeze(0) if label.ndim == 1 else label
    img = img.unsqueeze(0).to(device).requires_grad_(True)
    label = label.to(device)
    baseline = torch.zeros_like(img).to(device)

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

def load_and_preprocess_image(image):
    image = image.convert('RGB')
    image_tensor = test_transform(image)
    return image_tensor

st.title("Retinal Disease Classification with Integrated Gradients")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Load and preprocess the image
    test_img = load_and_preprocess_image(Image.open(uploaded_file))

    # Extract disease labels
    disease_labels = [label for label in labels_list[2:] if label not in ['HR', 'ODPM']]
    
    # Make predictions
    vit_model.eval()
    with torch.no_grad():
        output = vit_model(test_img.unsqueeze(0).to(device)).logits

    predicted_probs = torch.sigmoid(output).squeeze().cpu().numpy()
    prob_threshold = 0.3
    predicted_diseases = [disease_labels[i] for i in range(len(predicted_probs)) if predicted_probs[i] > prob_threshold]

    # Display results
    st.write(f"Predicted diseases: {predicted_diseases}")

    # Integrated Gradients
    true_labels_reshaped = torch.zeros(len(disease_labels)).to(device)
    attribution = integrated_gradients(vit_model, test_img, true_labels_reshaped)
    attribution = attribution.squeeze()

    threshold = 0.3
    important_areas = attribution > threshold
    kernel = np.ones((5, 5), np.uint8)
    dilated_areas = cv2.dilate(important_areas.astype(np.uint8), kernel, iterations=1)

    # Prepare for visualization
    original_image = test_img.permute(1, 2, 0).cpu().numpy()
    original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())
    original_image = original_image.astype(np.float32)

    black_mask = (original_image.sum(axis=2) > 0.1)
    highlight_color = np.array([1, 0, 1])
    highlighted_image = np.zeros((*dilated_areas.shape, 3))

    highlighted_image[(dilated_areas == 1) & (black_mask)] = highlight_color
    highlighted_image = highlighted_image.astype(np.float32)

    blended_image = cv2.addWeighted(original_image, 0.6, highlighted_image, 0.4, 0)

    # Display images
    st.subheader("Original Image")
    st.image(original_image, channels="RGB")

    st.subheader("Integrated Gradients Attribution")
    st.image(attribution, channels="GRAY", use_column_width=True)

    st.subheader("Highlighted Important Areas")
    st.image(blended_image, channels="RGB")

