import streamlit as st
from model import load_model
from utils import load_and_preprocess_image, integrated_gradients
import numpy as np

# Load the model
vit_model, device = load_model()

# Streamlit app
st.title("Retinal Disease Classification")
st.write("Upload a retinal image for disease classification:")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    image = load_and_preprocess_image(uploaded_file)

    # Convert the tensor to a NumPy array for display
    image_numpy = image.permute(1, 2, 0).detach().cpu().numpy()  # Change to HWC format

    # Scale the pixel values to [0, 255] if necessary (assuming values are in [0, 1])
    #image_numpy = (image_numpy * 255).astype(np.uint8)
    
    st.image(image_numpy, caption="Uploaded Image", use_column_width=True)
    
    disease_labels = ['DR', 'ARMD', 'MH', 'DN', 'MYA', 'BRVO', 'TSLN', 'ERM', 'LS', 'MS', 
                      'CSR', 'ODC', 'CRVO', 'TV', 'AH', 'ODP', 'ODE', 'ST', 'AION', 
                      'PT', 'RT', 'RS', 'CRS', 'EDN', 'RPEC', 'MHL', 'RP', 'CWS', 
                      'CB', 'PRH', 'MNF', 'HR', 'CRAO', 'TD', 'CME', 'PTCR', 
                      'CF', 'VH', 'MCA', 'VS', 'BRAO', 'PLQ', 'HPED', 'CL']

    # Make predictions
    vit_model.eval()
    with torch.no_grad():
        output = vit_model(image.unsqueeze(0).to(device)).logits

    predicted_probs = torch.sigmoid(output).squeeze().cpu().numpy()
    prob_threshold = 0.3
    predicted_diseases = [disease_labels[i] for i in range(len(predicted_probs)) if predicted_probs[i] > prob_threshold]

    st.write(f"Predicted diseases: {predicted_diseases}")

    # Integrated Gradients
    true_labels_reshaped = torch.zeros(len(disease_labels)).to(device)
    attribution = integrated_gradients(vit_model, image, true_labels_reshaped)
    attribution = attribution.squeeze()

    # Prepare images for display
    original_image = image.permute(1, 2, 0).cpu().numpy()
    original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())
    original_image = original_image.astype(np.float32)

    black_mask = (original_image.sum(axis=2) > 0.1)
    kernel = np.ones((5, 5), np.uint8)
    dilated_areas = cv2.dilate((attribution > 0.3).astype(np.uint8), kernel, iterations=1)
    
    highlight_color = np.array([1, 0, 1])
    highlighted_image = np.zeros((*dilated_areas.shape, 3))
    highlighted_image[(dilated_areas == 1) & (black_mask)] = highlight_color
    highlighted_image = highlighted_image.astype(np.float32)

    blended_image = cv2.addWeighted(original_image, 0.6, highlighted_image, 0.4, 0)

    # Display images
    st.write("### Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(original_image, caption="Original Image")
    
    with col2:
        st.image(attribution, caption="Integrated Gradients", use_column_width=True, channels="GRAY")
    
    with col3:
        st.image(blended_image, caption="Highlighted Areas", use_column_width=True)
