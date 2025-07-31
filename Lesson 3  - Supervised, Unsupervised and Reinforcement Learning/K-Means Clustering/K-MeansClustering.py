import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

def load_data(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)

def select_features(df, feature_cols):
    """Select relevant columns for clustering."""
    return df[feature_cols]

def scale_features(X):
    """Standardize the feature data."""
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler

def compute_inertia(X_scaled, k_range=range(1, 11)):
    """Compute inertia for K-Means clustering across a range of K values."""
    inertia = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertia.append(km.inertia_)
    return inertia

def plot_elbow_curve(inertia, k_range):
    """Plot the elbow method curve."""
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertia, marker='o')
    plt.title("Elbow Method: Finding Optimal Number of Clusters")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia (Within-Cluster Sum of Squares)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def perform_kmeans(X_scaled, n_clusters=5):
    """Perform KMeans clustering and return the model and labels."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    return kmeans, kmeans.labels_

def plot_clusters(df, x_col, y_col, cluster_col='Cluster'):
    """Visualize clusters using selected features."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=cluster_col, palette='tab10', s=80)
    plt.title(f"Mall Customers – K-Means Clustering (K={df[cluster_col].nunique()})")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def cluster_summary(df, cluster_col='Cluster'):
    """Print average values of each feature per cluster."""
    print("\n📋 Average Feature Values by Cluster:")
    print(df.groupby(cluster_col)[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean())

def main():
    base_path = os.path.dirname(__file__)  # directory of the script
    csv_path = os.path.join(base_path,"Mall_Customers.csv")
    # Step 1: Load and prepare data
    df = load_data(csv_path)
    features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    X = select_features(df, features)
    X_scaled, _ = scale_features(X)

    # Step 2: Elbow method
    k_range = range(1, 11)
    inertia = compute_inertia(X_scaled, k_range)
    plot_elbow_curve(inertia, k_range)

    # Step 3: Fit and apply KMeans
    kmeans, labels = perform_kmeans(X_scaled, n_clusters=5)
    df['Cluster'] = labels

    # Step 4: Visualization
    plot_clusters(df, 'Annual Income (k$)', 'Spending Score (1-100)', 'Cluster')

    # Step 5: Summary
    cluster_summary(df)

if __name__ == "__main__":
    main()
