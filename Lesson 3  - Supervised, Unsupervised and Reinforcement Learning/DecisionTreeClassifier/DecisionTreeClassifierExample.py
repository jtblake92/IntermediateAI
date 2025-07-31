import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def load_and_clean_data():
    """Load and clean the Titanic dataset."""
    df = sns.load_dataset('titanic')  # Replace with pd.read_csv('your_file.csv') if needed
    df_clean = df[['survived', 'pclass', 'sex', 'age', 'fare', 'embarked']].dropna()

    # Encode categorical variables
    df_clean['sex'] = df_clean['sex'].map({'male': 0, 'female': 1})
    df_clean['embarked'] = df_clean['embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    return df_clean


def split_data(df):
    """Split the dataset into train and test sets."""
    X = df.drop('survived', axis=1)
    y = df['survived']
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(X_train, y_train):
    """Train a Decision Tree model."""
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance and print metrics."""
    y_pred = model.predict(X_test)
    print(f"✅ Accuracy: {round(accuracy_score(y_test, y_pred), 2)}")
    print("📋 Classification Report:\n", classification_report(y_test, y_pred))

    return y_pred


def print_tree_rules(model, feature_names):
    """Display the tree structure in text format."""
    print("🧾 Decision Tree Rules (Text View):")
    print("-" * 50)
    print(export_text(model, feature_names=feature_names, show_weights=True))
    print("-" * 50)


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
