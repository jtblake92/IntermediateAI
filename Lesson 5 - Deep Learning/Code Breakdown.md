
# 🧥 Fashion Classifier with Feedback Loop

This project contains a Streamlit app powered by deep learning that allows users to classify clothing images using a pre-trained or fine-tuned MobileNetV2 model. Users can upload images, view predictions, and provide feedback, which is logged for retraining a custom model that evolves over time.

---

## 📁 Project Structure

```
├── app.py                      # Main Streamlit app for image classification
├── train_feedback_model.py    # Script to retrain model using feedback
├── feedback/
│   ├── feedback_log.csv       # Stores feedback records (image path, prediction, correct label)
│   └── *.jpg                  # Feedback image samples
├── custom_clothing_model.h5   # Trained custom model (after enough feedback)
├── class_labels.npy           # Class labels corresponding to the trained model
```

---

## 🚀 How It Works

### 🔍 `app.py`
- Loads either a default MobileNetV2 model or a custom-trained version (`custom_clothing_model.h5`) if available.
- Accepts user-uploaded clothing images.
- Predicts the clothing type and shows top prediction(s).
- Allows users to confirm or correct the prediction.
- Stores corrected labels and images in the `feedback/` folder for future retraining.

### 🧠 `train_feedback_model.py`
- Reads the `feedback/feedback_log.csv` and associated images.
- Preprocesses images and labels.
- Fine-tunes a MobileNetV2 model using transfer learning.
- Ensures that there are at least 2 images per class and at least 2 unique classes.
- Saves the new model (`custom_clothing_model.h5`) and corresponding class labels (`class_labels.npy`).

---

## 🔁 Feedback Loop

Every user interaction can help improve the AI:
- If the model is wrong, the corrected label is submitted and stored.
- When enough diverse feedback is collected, retrain using:

```bash
python train_feedback_model.py
```

This updates the app’s model for future predictions.

---

## 💡 Tips

- Add at least 2 samples per new label to allow the training script to work properly.
- Use the feedback folder as your growing training dataset.
- This app uses transfer learning for faster and more accurate results with fewer examples.

---

## 🧪 Example Usage

```bash
streamlit run app.py
```

Upload clothing images and provide feedback. Once feedback has been collected:

```bash
python train_feedback_model.py
```

The next time the app loads, it will use the updated model with your new classes.

---

## 🙋 Help Each Other

Use the discussion tab in the repo to:
- Ask questions
- Suggest new categories
- Share model accuracy tips
- Collaborate on feedback loops

---

## 📂 Notes

- The app uses TensorFlow + MobileNetV2 and Streamlit.
- The feedback system is lightweight and requires only image files + one CSV.

Happy training! 🧠✨
