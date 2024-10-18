import time  # For simulating progress updates

st.title("Retinal Disease Classification")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Progress bar
    progress_bar = st.progress(0)
    
    # Simulating the first step of image loading and processing
    progress_bar.progress(10)
    test_img = load_and_preprocess_image(image)

    disease_labels = [label for label in labels_list[2:] if label not in ['HR', 'ODPM']]

    vit_model.eval()

    # Simulating the second step of model prediction
    progress_bar.progress(30)
    with torch.no_grad():
        output = vit_model(test_img.unsqueeze(0).to(device)).logits

    predicted_probs = torch.sigmoid(output).squeeze().cpu().numpy()
    prob_threshold = 0.3
    predicted_diseases = [disease_labels[i] for i in range(len(predicted_probs)) if predicted_probs[i] > prob_threshold]

    st.write(f"Predicted diseases: {predicted_diseases}")

    predicted_full_list_names = [labels_dict[label] for label in predicted_diseases]
    st.write(f"Predicted Disease Full Names: {predicted_full_list_names}")

    # Simulating the third step of calculating attributions using integrated gradients
    progress_bar.progress(60)
    true_labels_reshaped = torch.zeros(len(disease_labels)).to(device)
    attribution = integrated_gradients(vit_model, test_img, true_labels_reshaped)
    attribution = attribution.squeeze()

    threshold = 0.3
    important_areas = attribution > threshold
    kernel = np.ones((5, 5), np.uint8)
    dilated_areas = cv2.dilate(important_areas.astype(np.uint8), kernel, iterations=1)

    # Simulating the fourth step of preparing the image for visualization
    progress_bar.progress(80)
    original_image = test_img.permute(1, 2, 0).cpu().numpy()
    original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())
    original_image = original_image.astype(np.float32)

    black_mask = (original_image.sum(axis=2) > 0.1)
    highlight_color = np.array([1, 0, 1], dtype=np.float32)
    highlighted_image = np.zeros((*dilated_areas.shape, 3), dtype=np.float32)
    highlighted_image[(dilated_areas == 1) & (black_mask)] = highlight_color

    blended_image = cv2.addWeighted(original_image, 0.6, highlighted_image, 0.4, 0)

    # Simulating the final step of plotting and displaying the results
    progress_bar.progress(100)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(original_image)
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    ax[1].imshow(attribution, cmap='hot', interpolation='nearest')
    ax[1].set_title("Integrated Gradients")
    ax[1].axis('off')

    ax[2].imshow(blended_image)
    ax[2].set_title(f"Highlighted Areas - Predicted Class: {predicted_diseases}")
    ax[2].axis('off')

    st.pyplot(fig)

    # Completion message
    st.success("Image processing complete!")
