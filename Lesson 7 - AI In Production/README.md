# 🧠 Customer Support Message Classifier

This is a beginner-friendly AI mini-project for classifying customer support messages into categories like billing, technical issues, or positive feedback.

## 🚀 How to Run

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Train the model:
```
python train_model.py
```

3. Start the API:
```
python app.py
```

4. Send predictions:
```
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "I was double charged on my invoice"}'
```

## 📁 Project Structure

- `train_model.py` — trains the model and saves it as `model_v1.pkl`
- `app.py` — serves predictions via Flask API
- `data/messages.csv` — sample dataset
- `monitoring.log` — records prediction history
- `models/model_v1.pkl` — trained model

## ✅ Checklist

- [x] Accuracy > 80%
- [x] All predictions logged
- [x] Easy to retrain and improve

