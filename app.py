import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import streamlit as st
from PIL import Image
from transformers import ViTForImageClassification
import torchvision.transforms.v2 as transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Label Full Form Dict
labels_dict = {
    'ID': 'ID',
    'Disease_Risk': 'Disease_Risk',
    'DR': 'Diabetic retinopathy',
    'ARMD': 'Age-related macular degeneration',
    'MH': 'Media haze',
    'DN': 'Drusen',
    'MYA': 'Myopia',
    'BRVO': 'Branch retinal vein occlusion',
    'TSLN': 'Tessellation',
    'ERM': 'Epiretinal membrane',
    'LS': 'Laser scars',
    'MS': 'Macular scars',
    'CSR': 'Central serous retinopathy',
    'ODC': 'Optic disc cupping',
    'CRVO': 'Central retinal vein occlusion',
    'TV': 'Tortuous vessels',
    'AH': 'Asteroid hyalosis',
    'ODP': 'Optic disc pallor',
    'ODE': 'Optic disc edema',
    'ST': 'Optociliary shunt',
    'AION': 'Anterior ischemic optic neuropathy',
    'PT': 'Parafoveal telangiectasia',
    'RT': 'Retinal traction',
    'RS': 'Retinitis',
    'CRS': 'Chorioretinitis',
    'EDN': 'Exudation',
    'RPEC': 'Retinal pigment epithelium changes',
    'MHL': 'Macular hole',
    'RP': 'Retinitis pigmentosa',
    'CWS': 'Cotton-wool spots',
    'CB': 'Coloboma',
    'ODPM': 'Optic disc pit maculopathy',
    'PRH': 'Preretinal hemorrhage',
    'MNF': 'Myelinated nerve fibers',
    'HR': 'Hemorrhagic retinopathy',
    'CRAO': 'Central retinal artery occlusion',
    'TD': 'Tilted disc',
    'CME': 'Cystoid macular edema',
    'PTCR': 'Post-traumatic choroidal rupture',
    'CF': 'Choroidal folds',
    'VH': 'Vitreous hemorrhage',
    'MCA': 'Macroaneurysm',
    'VS': 'Vasculitis',
    'BRAO': 'Branch retinal artery occlusion',
    'PLQ': 'Plaque',
    'HPED': 'Hemorrhagic pigment epithelial detachment',
    'CL': 'Collateral'
}

# Load labels
labels_list = ['ID', 'Disease_Risk', 'DR', 'ARMD', 'MH', 'DN', 'MYA', 'BRVO', 'TSLN',
               'ERM', 'LS', 'MS', 'CSR', 'ODC', 'CRVO', 'TV', 'AH', 'ODP', 'ODE', 'ST',
               'AION', 'PT', 'RT', 'RS', 'CRS', 'EDN', 'RPEC', 'MHL', 'RP', 'CWS',
               'CB', 'ODPM', 'PRH', 'MNF', 'HR', 'CRAO', 'TD', 'CME', 'PTCR', 'CF',
               'VH', 'MCA', 'VS', 'BRAO', 'PLQ', 'HPED', 'CL']

PRET_MEANS = [0.485, 0.456, 0.406]
PRET_STDS = [0.229, 0.224, 0.225]

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=PRET_MEANS, std=PRET_STDS)
])

# Load the saved model
vit_model = ViTForImageClassification.from_pretrained('mandarchaudharii/retinal_multiclass')
vit_model.to(device)

def integrated_gradients(model, img, label, steps=50):
    model.eval()
    label = label.unsqueeze(0) if label.ndim == 1 else label
    img = img.unsqueeze(0).to(device).requires_grad_(True)
    
    label = label.to(device)
    baseline = torch.zeros_like(img).to(device)
    scaled_images = [baseline + (float(i) / steps) * (img - baseline) for i in range(steps + 1)]

    z=0
    
    grads = []
    for scaled_img in scaled_images:
        output = model(scaled_img).logits
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output, label.float())
        model.zero_grad()
        loss.backward()
        grads.append(img.grad.data.cpu().numpy())

        z=z+1
        progress_bar.progress(int(z * progress_step * 100))  # Update progress

    progress_bar.progress(int(200 * progress_step * 100))  # Update progress
    
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

st.title("Retinal Disease Classification")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    test_img = load_and_preprocess_image(image)
    
    disease_labels = [label for label in labels_list[2:] if label not in ['HR', 'ODPM']]
    
    vit_model.eval()
    with torch.no_grad():
        output = vit_model(test_img.unsqueeze(0).to(device)).logits

    predicted_probs = torch.sigmoid(output).squeeze().cpu().numpy()
    prob_threshold = 0.3
    predicted_diseases = [(disease_labels[i], predicted_probs[i]) for i in range(len(predicted_probs)) if predicted_probs[i] > prob_threshold]
    
    if predicted_diseases:
        st.write("Predicted Diseases:")
        for disease, prob in predicted_diseases:
            full_name = labels_dict[disease]
            st.write(f"{disease} ({full_name}) - Probabilistic Score: {prob * 100:.2f}%")
    else:
        st.write("No diseases detected above the probability threshold.")

    st.write("Images Generation Progress.")
    progress_bar = st.progress(0)  # Initialize progress bar
    progress_step = 1 / 200

    true_labels_reshaped = torch.zeros(len(disease_labels)).to(device)
    
    attribution = integrated_gradients(vit_model, test_img, true_labels_reshaped)
    attribution = attribution.squeeze()

    threshold = 0.3
    important_areas = attribution > threshold
    kernel = np.ones((5, 5), np.uint8)
    dilated_areas = cv2.dilate(important_areas.astype(np.uint8), kernel, iterations=1)
  
    # Load and process the original image for visualization
    original_image = test_img.permute(1, 2, 0).cpu().numpy()
    original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())
    original_image = original_image.astype(np.float32)  # Ensure it's float32
    
    # Create a mask for non-black areas in the original image
    black_mask = (original_image.sum(axis=2) > 0.1)
    
    # Create a color mask for highlighting
    highlight_color = np.array([1, 0, 1], dtype=np.float32)  # Ensure it's float32
    highlighted_image = np.zeros((*dilated_areas.shape, 3), dtype=np.float32)  # Ensure it's float32
    
    # Apply the highlight color only to important areas that are also non-black
    highlighted_image[(dilated_areas == 1) & (black_mask)] = highlight_color
    
    # Blend the original image with the highlighted areas
    blended_image = cv2.addWeighted(original_image, 0.6, highlighted_image, 0.4, 0)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(original_image)
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    ax[1].imshow(attribution, cmap='hot', interpolation='nearest')
    ax[1].set_title("Integrated Gradients")
    ax[1].axis('off')

    ax[2].imshow(blended_image)
    ax[2].set_title(f"Highlighted Areas For Predicted Diseases")
    ax[2].axis('off')

    st.pyplot(fig)
