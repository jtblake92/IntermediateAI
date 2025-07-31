# 🛍️ Mall Customer Segmentation with K-Means Clustering

This project uses **K-Means Clustering** to segment mall customers based on features like **age**, **annual income**, and **spending score**. It demonstrates how unsupervised learning can uncover valuable customer groups for marketing and business strategy.

---

## 📂 Project Overview

The workflow includes:

1. Loading and selecting features from the dataset
2. Standardizing the data
3. Using the **Elbow Method** to choose the best number of clusters
4. Applying K-Means clustering
5. Visualising the clusters
6. Summarising the key patterns

---

## 📊 Dataset

The dataset `Mall_Customers.csv` should include:

- `Age`
- `Annual Income (k$)`
- `Spending Score (1-100)`

These features will be used to group customers based on similar purchasing behaviors.

---

## 🔁 Step-by-Step Breakdown

### 1. Load and Preprocess the Data

```python
df = pd.read_csv("Mall_Customers.csv")
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
X_scaled = StandardScaler().fit_transform(X)
```

We select three numeric features and standardise them to ensure fair distance-based clustering.

---

### 2. Use the Elbow Method

```python
inertia = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)
```

The **inertia** shows how tight each cluster is. We plot this to find the "elbow point" — where adding more clusters stops giving big improvements.

```python
plt.plot(range(1, 11), inertia, marker='o')
plt.title("Elbow Method")
```

---

### 3. Apply K-Means Clustering

```python
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)
```

We choose **5 clusters** based on the elbow result and assign each customer to one.

---

### 4. Visualise Clusters

```python
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster')
```

This plot helps you **see how customers group** by spending vs. income. You can explore other feature combinations too.

---

### 5. Summarise Cluster Characteristics

```python
df.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
```

Prints the average profile of each cluster — useful for marketing personas like:

- 🎯 High-income, low-spending customers
- 🛍️ Young, high-spending, moderate-income customers
- 💰 Budget-conscious but loyal shoppers

---

## ✅ Summary

- K-Means helps identify **patterns in unlabelled data**.
- We used the **Elbow Method** to determine the optimal number of clusters.
- Visualisation reveals how **spending habits relate to income and age**.
- Businesses can use this to target promotions and personalise offers.

---

## 📦 Optional Extensions

- Try other cluster counts (e.g. 3, 6, 8)
- Add `Gender`, `CustomerID` as metadata
- Use PCA or t-SNE for deeper dimensionality reduction

> 🧠 Unsupervised learning like this is essential when there's no ground truth — let the data speak for itself!