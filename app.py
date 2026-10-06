
from pathlib import Path

import streamlit as st
from PIL import Image

from src.predict import predict_image
from src.gradcam import generate_gradcam


st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🔎",
    layout="wide"
)

st.title("Deepfake Detection System")
st.write(
    "Upload a face image to obtain a REAL/FAKE prediction "
    "and a Grad-CAM visualization."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded image")
    st.image(image, caption="Input image", width=400)

    with st.spinner("Detecting face and analyzing image..."):
        result = predict_image(image)

    if "error" in result:
        st.warning(result["error"])
    else:
        label = result["label"]
        confidence = result["confidence"]

        st.subheader("Prediction")

        if label == "FAKE":
            st.error(f"Prediction: {label}")
        else:
            st.success(f"Prediction: {label}")

        st.metric(
            "Model confidence",
            f"{confidence * 100:.2f}%"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Detected face")
            st.image(
                result["face_image"],
                caption="Face extracted by MTCNN",
                width=300
            )

        with col2:
            st.subheader("Grad-CAM heatmap")

            with st.spinner("Generating heatmap..."):
                face, overlay = generate_gradcam(image)

            if overlay is not None:
                st.image(
                    overlay,
                    caption="Highlighted regions influencing the model",
                    width=300
                )
            else:
                st.warning("Could not generate the heatmap.")

        st.caption(
            "Note: This is a model prediction, not proof that an image "
            "is authentic or manipulated. Grad-CAM highlights influential "
            "regions and does not independently verify a forgery."
        )
