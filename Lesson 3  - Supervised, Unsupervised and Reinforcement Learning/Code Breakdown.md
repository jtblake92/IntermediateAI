# 🧠 Decision Tree Titanic Classifier: Code Walkthrough with Narration

## ✳️ `load_and_clean_data`

> We start by loading the Titanic dataset and cleaning it up — removing missing values and converting text columns like 'sex' and 'embarked' into numbers so the model can understand them.

```python
def load_and_clean_data():
    """Load and clean the Titanic dataset."""
    df = sns.load_dataset('titanic')  # Replace with pd.read_csv('your_file.csv') if needed
    df_clean = df[['survived', 'pclass', 'sex', 'age', 'fare', 'embarked']].dropna()

    # Encode categorical variables
    df_clean['sex'] = df_clean['sex'].map({'male': 0, 'female': 1})
    df_clean['embarked'] = df_clean['embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    return df_clean
```

## ✳️ `split_data`

> Next, we split the data into training and test sets. This helps us build the model using one set of data, and then test how well it performs on new, unseen data.

```python
def split_data(df):
    """Split the dataset into train and test sets."""
    X = df.drop('survived', axis=1)
    y = df['survived']
    return train_test_split(X, y, test_size=0.2, random_state=42)
```

## ✳️ `train_model`

> We now train a Decision Tree with a maximum depth of 4. This keeps it from getting too complex and overfitting — basically, memorising the training data instead of learning patterns.

```python
def train_model(X_train, y_train):
    """Train a Decision Tree model."""
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    return model
```

## ✳️ `evaluate_model`

> The model makes predictions on the test set. We then check how accurate it was and look at a classification report that breaks down precision, recall, and F1 score for each class.

```python
def evaluate_model(model, X_test, y_test):
    """Evaluate model performance and print metrics."""
    y_pred = model.predict(X_test)
    print(f"✅ Accuracy: {round(accuracy_score(y_test, y_pred), 2)}")
    print("📋 Classification Report:\n", classification_report(y_test, y_pred))

    return y_pred
```

## ✳️ `print_tree_rules`

> To help you understand how the model makes decisions, we print a text version of the decision tree rules. It shows which features matter and how the model reaches a prediction.

```python
def print_tree_rules(model, feature_names):
    """Display the tree structure in text format."""
    print("🧾 Decision Tree Rules (Text View):")
    print("-" * 50)
    print(export_text(model, feature_names=feature_names, show_weights=True))
    print("-" * 50)
```

## ✳️ `plot_results`

> Now we visualise the model’s results: a confusion matrix shows how many predictions were right or wrong, a count plot shows survival predictions overall, and finally, we draw the full decision tree.

```python
def plot_results(y_test, y_pred, model, feature_names):
    """Plot confusion matrix, prediction distribution, and tree structure."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # Confusion Matrix
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax[0])
    ax[0].set_title("Confusion Matrix")
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("Actual")

    # Prediction Distribution
    y_labels = ['Did Not Survive' if val == 0 else 'Survived' for val in y_pred]
    sns.countplot(x=y_labels, hue=y_labels, palette='Set2', ax=ax[1], legend=False)
    ax[1].set_title("Prediction Distribution")
    ax[1].set_ylabel("Count")

    plt.tight_layout()
    plt.show()

    # Plot Tree Structure
    plt.figure(figsize=(18, 8))
    plot_tree(model, feature_names=feature_names, class_names=["Did Not Survive", "Survived"],
              filled=True, rounded=True)
    plt.title("Decision Tree Structure")
    plt.show()
```

## ✳️ `main`

> This is the main pipeline that runs all the steps: from loading data to training, evaluating, and displaying the model and results.

```python
def main():
    """Run the Titanic Decision Tree classification pipeline."""
    df = load_and_clean_data()
    X_train, X_test, y_train, y_test = split_data(df)

    model = train_model(X_train, y_train)
    y_pred = evaluate_model(model, X_test, y_test)

    print_tree_rules(model, feature_names=list(df.drop('survived', axis=1).columns))
    plot_results(y_test, y_pred, model, feature_names=list(df.drop('survived', axis=1).columns))


if __name__ == "__main__":
    main()
```
