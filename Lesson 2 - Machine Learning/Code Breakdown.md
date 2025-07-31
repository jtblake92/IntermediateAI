# 🌸 Iris Dataset Classification with Multiple Models

This guide walks through how to train, evaluate, and visualise a classification model using the classic **Iris dataset**. We'll use three models: **Random Forest**, **Logistic Regression**, and **K-Nearest Neighbours**. We'll focus our evaluations and visualisations on the **Random Forest** output.

---

## 🧰 Step 1: Import Required Libraries

We use `scikit-learn` for datasets, models, and evaluation tools, and `matplotlib`/`seaborn` for visualisations.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
```

---

## 🌼 Step 2: Load the Iris Dataset

We load the classic **Iris flower classification** dataset and extract features (`X`) and labels (`y`).

```python
iris = load_iris()
X, y = iris.data, iris.target
```

---

## 🧪 Step 3: Split the Data

Split the dataset into 80% training and 20% testing.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## 🧠 Step 4: Train Multiple Classifiers

We train three classifiers:

- **Random Forest** – ensemble of decision trees
- **Logistic Regression** – good for linear decision boundaries
- **KNN** – classifies based on nearest neighbours

```python
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

model1 = LogisticRegression(max_iter=200, random_state=42)
model1.fit(X_train, y_train)

model2 = KNeighborsClassifier(n_neighbors=3)
model2.fit(X_train, y_train)
```

---

## 🔍 Step 5: Predict and Evaluate

We make predictions using Random Forest and evaluate accuracy and classification report.

```python
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {round(accuracy, 2)}")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

---

## 📊 Step 6: Visualise Results

We visualise:

- Distribution of predicted classes
- Confusion matrix
- Feature importance from the Random Forest

```python
# Prediction counts
unique, counts = np.unique(y_pred, return_counts=True)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Feature importance
importances = model.feature_importances_
features = iris.feature_names

# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# 1. Prediction Distribution
axs[0].bar(iris.target_names[unique], counts, color='skyblue')
axs[0].set_title('Predicted Class Distribution')

# 2. Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names,
            ax=axs[1])
axs[1].set_title('Confusion Matrix')

# 3. Feature Importance
axs[2].barh(features, importances, color='mediumseagreen')
axs[2].set_title("Feature Importance")

plt.tight_layout()
plt.show()
```

---

## ✅ Summary

- Random Forest gave an accurate and interpretable result on the Iris dataset.
- We also trained Logistic Regression and KNN for comparison.
- Visualisations help us understand the model’s predictions, errors, and most important features.

🎯 Try changing models and parameters to see how performance changes!