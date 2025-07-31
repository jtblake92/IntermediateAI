# Import required libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train a Random Forest classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
model1 = LogisticRegression(max_iter=200, random_state=42)
model1.fit(X_train, y_train)
model2 = KNeighborsClassifier(n_neighbors=3)
model2.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {round(accuracy, 2)}\n")

print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Count predictions per class
unique, counts = np.unique(y_pred, return_counts=True)

# Get feature importances
importances = model.feature_importances_
features = iris.feature_names

# Set up subplots for visualisation
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Plot predicted class distribution
axs[0].bar(iris.target_names[unique], counts, color='skyblue')
axs[0].set_title('Predicted Class Distribution')
axs[0].set_xlabel('Iris Class')
axs[0].set_ylabel('Number of Predictions')

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names,
            ax=axs[1])
axs[1].set_title('Confusion Matrix')
axs[1].set_xlabel('Predicted')
axs[1].set_ylabel('Actual')

# Plot feature importance
axs[2].barh(features, importances, color='mediumseagreen')
axs[2].set_title("Feature Importance")
axs[2].set_xlabel("Importance Score")

# Adjust layout and display plots
plt.tight_layout()
plt.show()
