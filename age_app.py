import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


st.set_page_config(
    page_title="Age Prediction App",
    page_icon="🎂",
    layout="centered"
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "utkface_age_model.keras"
    )

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()


age_groups = [
    "0–12 years",
    "13–19 years",
    "20–35 years",
    "36–60 years",
    "61+ years"
]


st.title("🎂 Age Prediction System")

st.write(
    "Upload a face image or use your webcam "
    "to predict its age group using a trained CNN."
)

st.divider()


input_method = st.radio(
    "Choose how to provide a face image:",
    ["Upload Image", "Use Webcam"],
    horizontal=True
)

image = None

# -----------------------------
# Option 1: Upload image
# -----------------------------
if input_method == "Upload Image":

    uploaded_file = st.file_uploader(
        "Choose a face image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")


elif input_method == "Use Webcam":

    camera_photo = st.camera_input(
        "Take a picture using your webcam"
    )

    if camera_photo is not None:
        image = Image.open(camera_photo).convert("RGB")


if image is not None:

    st.image(
        image,
        caption="Selected face image",
        width=300
    )

    
    processed_image = image.convert("L")
    processed_image = processed_image.resize((48, 48))

    image_array = np.array(
        processed_image,
        dtype=np.float32
    ) / 255.0

    image_array = np.expand_dims(
        image_array, axis=-1
    )

    image_array = np.expand_dims(
        image_array, axis=0
    )

    
    if st.button(
        "Predict Age Group",
        type="primary"
    ):

        try:
            with st.spinner("Analyzing image..."):

                predictions = model.predict(
                    image_array,
                    verbose=0
                )

            predicted_class = int(
                np.argmax(predictions[0])
            )

            confidence = float(
                predictions[0][predicted_class]
            ) * 100

            st.success("Prediction completed!")

            st.subheader("Predicted Age Group")

            st.markdown(
                f"## 🎯 {age_groups[predicted_class]}"
            )

            

            

            
            st.subheader("Scores for Each Age Group")

            score_data = {
                "Age group": age_groups,
                "Score (%)": (
                    predictions[0] * 100
                ).round(2)
            }

            st.bar_chart(
                score_data,
                x="Age group",
                y="Score (%)"
            )

        except Exception as e:
            st.error(
                f"Could not process this image: {e}"
            )

st.divider()

st.caption(
    "UTKFace Age Classification | CNN Deep Learning Project"
)
