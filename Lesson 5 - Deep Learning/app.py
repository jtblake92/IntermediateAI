import streamlit as st
import numpy as np
from PIL import Image
from datetime import datetime
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

def load_default_model():
    from tensorflow.keras.applications import MobileNetV2
    return MobileNetV2(weights='imagenet')

def preprocess_uploaded_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

@st.cache_resource
def load_models():
    try:
        model = load_model("custom_clothing_model.h5")
        class_labels = np.load("class_labels.npy", allow_pickle=True)
        return model, class_labels.tolist(), True
    except:
        return load_default_model(), None, False

def predict(img_array, model, custom, class_labels):
    preds = model.predict(img_array)
    if custom:
        top_idx = np.argmax(preds)
        top_label = class_labels[top_idx]
        confidence = preds[0][top_idx]
        return [(top_label, confidence)]
    else:
        from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
        decoded = decode_predictions(preds, top=3)[0]
        return [(label, prob) for (_, label, prob) in decoded]

def save_feedback(image_file, predicted, correct):
    os.makedirs("feedback", exist_ok=True)
    csv_path = "feedback/feedback_log.csv"

    # Ensure feedback_log.csv exists
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            f.write("")  # Create empty file

    # Save the feedback image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img = Image.open(image_file)
    label = correct.replace(" ", "_").lower()
    filename = f"feedback/{label}_{timestamp}.jpg"
    img.save(filename)

    # Append feedback to CSV
    with open(csv_path, "a") as f:
        f.write(f"{filename},{predicted},{correct}\n")

st.set_page_config(page_title="🧠 Fashion Classifier", layout="centered")
st.title("🧥👟 Fashion Classifier with Feedback Learning")
st.write("Upload an image of clothing. The model will try to identify it, and you can give feedback to improve it!")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    img_array = preprocess_uploaded_image(uploaded_file)
    model, class_labels, is_custom = load_models()

    predictions = predict(img_array, model, is_custom, class_labels)
    st.markdown("### 🧠 Prediction:")
    for i, (label, confidence) in enumerate(predictions):
        st.write(f"{i+1}. **{label}** — {confidence*100:.2f}%")

    st.markdown("### 💬 Feedback")
    correct = st.radio("Was the top prediction correct?", ("Yes", "No"))
    if correct == "No":
        correct_label = st.text_input("What should the correct label be?")
        if correct_label and st.button("Submit Correction"):
            save_feedback(uploaded_file, predictions[0][0], correct_label)
            st.success("✅ Feedback recorded!")
    else:
        if st.button("Confirm Prediction"):
            save_feedback(uploaded_file, predictions[0][0], predictions[0][0])
            st.success("👍 Thanks for confirming!")