import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

ShootingData = pd.read_excel('/content/fatal-police-shootings-data.xlsx')
ShootingAgencies = pd.read_excel('/content/fatal-police-shootings-agencies.xlsx')

ShootingData.shape
ShootingAgencies.shape

ShootingData.columns
ShootingAgencies.columns

ShootingData.count()

ShootingData.dtypes
ShootingAgencies.dtypes

ShootingAgencies.columns

# Perform one-hot encoding for the "race" column in ShootingData
ShootingData_encoded = pd.get_dummies(ShootingData, columns=['race'], prefix='Race')

# Drop the original "race" column
ShootingData = ShootingData.drop(columns=['race'])

# Concatenate the encoded columns with the original DataFrame
ShootingData = pd.concat([ShootingData, ShootingData_encoded], axis=1)

ShootingData.columns

merge_data = pd.merge(ShootingData, ShootingAgencies, on=['id'], how='inner', suffixes=('_left', '_right'))

merge_data['age'].fillna(merge_data['age'].mean(), inplace=True)
merge_data['latitude'].fillna(merge_data['latitude'].mean(), inplace=True)
merge_data['longitude'].fillna(merge_data['longitude'].mean(), inplace=True)

merge_data.shape

from typing_extensions import Final
plt.figure(figsize=(10, 6))
ax = merge_data.groupby(['race', 'id']).size().unstack().plot(kind='bar', stacked=True)
plt.title('Distribution of Case ID by Race')
plt.xlabel('Race')
plt.ylabel('Count')
plt.xticks(rotation=0)  # Rotate x-axis labels for better visibility
plt.legend(title='Case ID')
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(merge_data['date'], bins=5, edgecolor='black', alpha=0.7, rwidth=0.85)
plt.title('Histogram of Case ID by Date')
plt.xlabel('Date')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

X = merge_data[['latitude', 'longitude']]

num_clusters = 4

kmeans = KMeans(n_clusters=num_clusters)
merge_data['Cluster'] = kmeans.fit_predict(X)

plt.scatter(merge_data['longitude'], merge_data['latitude'], c=merge_data['Cluster'], cmap='viridis')
plt.title('K-Means Clustering of Latitude and Longitude')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

incident_counts = ShootingData.groupby(['latitude', 'longitude'])['id'].count().reset_index()

plt.figure(figsize=(10, 8))
sns.heatmap(data=incident_counts.pivot('latitude', 'longitude', 'id'), cmap='YlOrRd', annot=True, fmt='d', linewidths=.5)
plt.title('Shooting Incidents by Latitude and Longitude')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()