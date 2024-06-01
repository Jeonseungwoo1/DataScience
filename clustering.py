import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np

# Load the dataframe
df = pd.read_csv('Video_Games_Sales_as_at_22_Dec_2016.csv')

# Select necessary columns
df = df[['Platform', 'Genre', 'Global_Sales', 'User_Score', 'Critic_Score']]

# Handle missing values
df = df.dropna()

# Convert User_Score and Critic_Score to float
df['User_Score'] = pd.to_numeric(df['User_Score'], errors='coerce')
df['Critic_Score'] = pd.to_numeric(df['Critic_Score'], errors='coerce')

# Handle missing values
df = df.dropna()

# Feature selection and scaling, encoding
numeric_features = ['Global_Sales', 'User_Score', 'Critic_Score']
categorical_features = ['Platform', 'Genre']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Transform the data using the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
features_scaled = pipeline.fit_transform(df)

# K-Means clustering pipeline
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(features_scaled)

# Visualize clustering results (2D projection)
plt.figure(figsize=(15,8))
sns.scatterplot(data=df, x='Global_Sales', y='User_Score', hue='Cluster', palette='tab10', s=50, edgecolor='w', alpha=0.75)
plt.title('K-Means Clustering of Video Games')
plt.xlabel('Global Sales')
plt.ylabel('User Score')
plt.legend(title='Cluster')
plt.show()

plt.figure(figsize=(15,8))
sns.scatterplot(data=df, x='Global_Sales', y='Critic_Score', hue='Cluster', palette='tab10', s=50, edgecolor='w', alpha=0.75)
plt.title('K-Means Clustering of Video Games')
plt.xlabel('Global Sales')
plt.ylabel('Critic Score')
plt.legend(title='Cluster')
plt.show()

# Analyze characteristics of each cluster
for cluster in df['Cluster'].unique():
    print(f"Cluster {cluster}")
    print(df[df['Cluster'] == cluster][['Platform', 'Genre', 'Global_Sales', 'User_Score', 'Critic_Score']].describe())
    print("\n")

# Visualize platform distribution by cluster and total sales by platform
plt.figure(figsize=(14, 10))
sns.countplot(data=df, x='Platform', hue='Cluster', palette='tab10')
platform_sales = df.groupby('Platform')['Global_Sales'].sum().reset_index()
plt.twinx()
sns.lineplot(data=platform_sales, x='Platform', y='Global_Sales', color='black', marker='o')
plt.title('Platform Distribution by Cluster and Total Sales')
plt.xlabel('Platform')
plt.ylabel('Count')
plt.legend(title='Cluster')
plt.xticks(rotation=45)
plt.show()

# Visualize genre distribution by cluster and total sales by genre
plt.figure(figsize=(14, 10))
sns.countplot(data=df, x='Genre', hue='Cluster', palette='tab10')
genre_sales = df.groupby('Genre')['Global_Sales'].sum().reset_index()
plt.twinx()
sns.lineplot(data=genre_sales, x='Genre', y='Global_Sales', color='black', marker='o')
plt.title('Genre Distribution by Cluster and Total Sales')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.legend(title='Cluster')
plt.xticks(rotation=45)
plt.show()

# Calculate silhouette score
silhouette_avg = silhouette_score(df[numeric_features], df['Cluster'])
print(f'Silhouette Score: {silhouette_avg:.2f}')

# Function to visualize silhouette scores
def plot_silhouette(n_clusters, features_scaled):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features_scaled)
    silhouette_avg = silhouette_score(features_scaled, cluster_labels)
    sample_silhouette_values = silhouette_samples(features_scaled, cluster_labels)
    
    y_lower = 10
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    
    ax.set_title(f"Silhouette plot for {n_clusters} clusters")
    ax.set_xlabel("Silhouette coefficient values")
    ax.set_ylabel("Cluster label")
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_yticks([])
    ax.set_xticks(np.linspace(-0.1, 1, 11))
    plt.show()

# Visualize silhouette scores for cluster numbers 2, 3, 4, 5
for n_clusters in [2, 3, 4, 5]:
    plot_silhouette(n_clusters, features_scaled)

# Elbow Method for optimal number of clusters
ssd = []
range_n_clusters = range(1, 11)

for num_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(features_scaled)
    ssd.append(kmeans.inertia_)

# Plot the SSD for each number of clusters
plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, ssd, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Squared Distances (Inertia)')
plt.title('Elbow Method For Optimal Number of Clusters')
plt.show()
