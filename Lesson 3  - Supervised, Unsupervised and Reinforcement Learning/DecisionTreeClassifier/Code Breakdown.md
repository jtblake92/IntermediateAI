# 🚢 Titanic Survival Prediction – Decision Tree Classifier

This project demonstrates how to build and visualise a **Decision Tree** model to predict survival on the Titanic. The process includes data cleaning, model training, evaluation, and visualisation.

---

## 🗂 Dataset Overview

We use the Titanic dataset from the `seaborn` library. It contains information about passengers, such as age, fare, class, and whether they survived.

---

## 🔧 Workflow Summary

### 1. Load and Clean the Data

```python
df = sns.load_dataset('titanic')
df_clean = df[['survived', 'pclass', 'sex', 'age', 'fare', 'embarked']].dropna()

# Encode categorical columns
df_clean['sex'] = df_clean['sex'].map({'male': 0, 'female': 1})
df_clean['embarked'] = df_clean['embarked'].map({'S': 0, 'C': 1, 'Q': 2})
```

We retain key features and encode categorical values for machine learning.

---

### 2. Split the Dataset

```python
X = df_clean.drop('survived', axis=1)
y = df_clean['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Split into 80% training and 20% test data.

---

### 3. Train the Decision Tree Model

```python
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)
```

We limit `max_depth` to prevent overfitting and help with interpretability.

---

### 4. Evaluate the Model

```python
y_pred = model.predict(X_test)
accuracy_score(y_test, y_pred)
classification_report(y_test, y_pred)
```

Prints overall accuracy and a class-by-class breakdown of precision, recall, and F1-score.

---

### 5. Visualise the Tree & Results

#### Confusion Matrix and Prediction Distribution

```python
# Heatmap of true vs predicted
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)

# Count plot of predicted labels
sns.countplot(x=['Did Not Survive' if val == 0 else 'Survived' for val in y_pred])
```

#### Tree Logic (Text View)

```python
export_text(model, feature_names=list(X.columns), show_weights=True)
```

#### Tree Diagram

```python
plot_tree(model, feature_names=X.columns, class_names=["Did Not Survive", "Survived"], filled=True)
```

---

## ✅ Summary

- We built a Decision Tree to classify Titanic survival outcomes.
- Cleaned and encoded the data.
- Evaluated accuracy and visualised results.
- Interpreted decision logic via text and visual format.

This exercise helps solidify how decision trees work and how they can be interpreted for real-world datasets.

🎯 Try experimenting by:
- Increasing tree depth
- Adding more features like `sibsp` and `parch`
- Using a different classifier like `RandomForestClassifier`