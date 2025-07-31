
# 🧠 Customer Support Message Classifier (Mini Project)

A beginner-friendly machine learning project that builds a text classifier to categorize customer support messages into labels like **billing**, **technical**, or **feedback**.

---

## 🗂️ Project Structure

```
.
├── app.py                  # Flask API to serve predictions
├── train_model.py          # Model training and saving script
├── requirements.txt        # Dependencies
├── monitoring.log          # Auto-generated prediction logs
├── models/
│   └── model_v1.pkl        # Saved vectorizer and model
├── data/
│   └── messages.csv        # Input dataset (message + category)
└── README.md               # Original guide
```

---

## 🚀 How to Run It

### 🔧 Step 1: Install Required Packages

```bash
pip install -r requirements.txt
```

### 🧠 Step 2: Train the Model

This loads data from `data/messages.csv`, vectorizes it using TF-IDF, trains a logistic regression classifier, prints accuracy, and saves the model to `models/model_v1.pkl`.

```bash
python train_model.py
```

Expected output:
```
✅ Accuracy: 90.00%
✅ Model saved to models/model_v1.pkl
```

### 🌐 Step 3: Start the API Server

```bash
python app.py
```

You’ll see:
```
* Running on http://127.0.0.1:5000 or similar
```

### 📡 Step 4: Make Predictions

Send a POST request to classify a support message:

```bash
curl -X POST http://localhost:5000/predict   -H "Content-Type: application/json"   -d '{"message": "I was double charged on my invoice"}'
```

Response:
```json
{ "category": "billing" }
```

---

## 📝 Logging and Monitoring

- Every prediction is automatically logged in `monitoring.log`.
- Format:  
  `2025-07-31T14:02:03 | input: I want to update my billing address. | prediction: billing`

This allows for later auditing, error analysis, or retraining decisions.

---

## 📚 Tech Used

- Python
- Flask (for API)
- Scikit-learn (TF-IDF, Logistic Regression)
- Pandas (data loading)
- Pickle (model saving)
- CSV for input and logging

---

## 📌 Tips

- You can update `messages.csv` with more examples and retrain the model.
- The `.pkl` file stores both the trained vectorizer and classifier together.
- For deployment, consider switching from Flask's development server to a production-grade WSGI server like Gunicorn.

---

## 🤝 Learning Together

Use your GitHub **Discussions tab** to:
- Share ideas for improvements
- Ask questions about NLP or Flask
- Contribute new message categories or datasets

---

## 📂 Reminder

Always `cd` into the project folder before running scripts:

```bash
cd your_project_directory
```

Then proceed with `python train_model.py` or `python app.py`.

---

Happy Building! 💬📊
