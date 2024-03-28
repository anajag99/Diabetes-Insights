import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
from folium.plugins import HeatMap
!pip install esda
!pip install libpysal
import esda
import libpysal
from libpysal.weights import KNN
import geopandas as gpd
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

ShootingData = pd.read_excel('/content/fatal-police-shootings-data.xlsx')

ShootingData.shape

ShootingData.loc[ShootingData['latitude'].isna()]

ShootingData[ShootingData['latitude'].isna()]

ShootingData["city2"] = ShootingData["city"] + " , " + ShootingData["state"]
ShootingData.drop(["state", "city"], axis = 1, inplace = True)
ShootingData.head()

ShootingData["city"] = ShootingData["city2"]
ShootingData.drop(["city2"], axis = 1, inplace = True)
ShootingData.head()

city_groups = ShootingData.groupby('city')

city_means = city_groups[['latitude','longitude']].transform('mean')

city_means

ShootingData['latitude'].fillna(city_means['latitude'],inplace = True)
ShootingData['longitude'].fillna(city_means['longitude'],inplace = True)

# Define the ordinal mapping
race_mapping = {'W': 1, 'B': 2, 'A': 3, 'H': 4, 'BH' : 5, 'O': 6, 'N' : 7}

# Apply ordinal encoding to 'Race' column
ShootingData['race'] = ShootingData['race'].map(race_mapping)
ShootingData['race'].fillna(-1, inplace=True)

gender_mapping = {'male': 1, 'female': 2, 'non-binary': 3}
ShootingData['gender'] = ShootingData['gender'].map(gender_mapping)
ShootingData['race'].fillna('Unknown', inplace=True)

# Mean imputation for latitude
mean_latitude = ShootingData['latitude'].median()
ShootingData['latitude'].fillna(mean_latitude, inplace=True)

# Mean imputation for longitude
mean_longitude = ShootingData['longitude'].median()
ShootingData['longitude'].fillna(mean_longitude, inplace=True)

# Mean imputation for age
mean_age = ShootingData['age'].median()
ShootingData['age'].fillna(mean_age, inplace=True)

ShootingData['threat_type'].fillna('unrecognized', inplace=True)

location_counts = ShootingData.groupby(['latitude', 'longitude'])['id'].count().reset_index()

# Display the locations and the count of occurrences
print(location_counts)

# Create a map centered around the mean latitude and longitude
map_shooting_heatmap = folium.Map(location=[location_counts['latitude'].mean(), location_counts['longitude'].mean()], zoom_start=5)

# Create a HeatMap layer based on the latitude and longitude coordinates and the count of occurrences
heat_data = [[row['latitude'], row['longitude'], row['id']] for index, row in location_counts.iterrows()]
HeatMap(heat_data).add_to(map_shooting_heatmap)

# Find the maximum count value in the heatmap data
max_count = max([row[2] for row in heat_data])

# Identify the locations with the highest incident density
hotspot_locations = [(row[0], row[1]) for row in heat_data if row[2] == max_count]

print("Maximum Count:", max_count)
print("Hotspot Locations:", hotspot_locations)

hotspot_filter = ShootingData[ShootingData.apply(lambda row: (row['latitude'], row['longitude']) in hotspot_locations, axis=1)]

# Analyze the race distribution in the hotspot locations
race_distribution = hotspot_filter['race'].value_counts()
states_in_hotspots = hotspot_filter['city'].unique()

# Print the race distribution in the hotspot locations
print("Race distribution in hotspot locations:")
print(race_distribution)
print("States associated with hotspot locations:")
print(states_in_hotspots)

# Create a GeoDataFrame from latitude and longitude columns
gdf = gpd.GeoDataFrame(ShootingData,
                      geometry=gpd.points_from_xy(ShootingData.longitude, ShootingData.latitude))

# Set the Coordinate Reference System (CRS)
gdf.crs = 'epsg:4326'

# Create a spatial weights matrix using k-nearest neighbors
w = KNN.from_dataframe(gdf, k=5)

# Extract the variable of interest (e.g., police shooting incidents)
incidents = gdf['id']

# Calculate Moran's I
moran = esda.Moran(incidents, w)

# Print the Moran's I statistic
print(f"Moran's I: {moran.I:.4f}")

# Check for significance
print(f"P-Value: {moran.p_sim:.4f}")

feature_columns = ['age', 'gender', 'latitude', 'longitude', 'id']
target_column = 'race'

X = ShootingData[feature_columns]
y = ShootingData[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = ['age', 'latitude', 'longitude']
categorical_features = ['gender']
numeric_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')

X_train[numeric_features] = numeric_imputer.fit_transform(X_train[numeric_features])
X_train[categorical_features] = categorical_imputer.fit_transform(X_train[categorical_features])

X_test[numeric_features] = numeric_imputer.fit_transform(X_test[numeric_features])
X_test[categorical_features] = categorical_imputer.fit_transform(X_test[categorical_features])

k = 100  # Number of neighbors to consider
knn = KNeighborsClassifier(n_neighbors=k)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)

# Define the hyperparameter grid
param_grid = {
    'n_neighbors': [3, 5, 7, 9],  # Number of neighbors to consider
    'weights': ['uniform', 'distance'],  # Weighting of neighbors (uniform or by distance)
    'p': [1, 2]  # Minkowski distance metric (1 for Manhattan, 2 for Euclidean)
}

knn = KNeighborsClassifier()

# Perform grid search with cross-validation
grid_search = GridSearchCV(knn, param_grid, scoring='accuracy', cv=5)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Train a final KNN model with the best hyperparameters
final_knn = KNeighborsClassifier(**best_params)
final_knn.fit(X_train, y_train)

# Evaluate the final KNN model on the test set
y_pred = final_knn.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

# Print the best hyperparameters
print("Best Hyperparameters:")
print(best_params)

# Print the test accuracy
print("Test Accuracy:", test_accuracy)

# Replace NaN values with mean (you can choose other strategies)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Create a K-Nearest Neighbors (KNN) classifier
knn_classifier = KNeighborsClassifier(n_neighbors=9, p=1, weights='uniform')

# Perform 5-fold cross-validation
cv_scores = cross_val_score(knn_classifier, X_imputed, y, cv=5, scoring='accuracy')

# Print the cross-validation scores
print("Cross-Validation Scores:", cv_scores)

# Calculate and print the mean accuracy
mean_accuracy = np.mean(cv_scores)
print("Mean Accuracy:", mean_accuracy)

# Create a Linear SVM classifier
svm_classifier = LinearSVC()

# Train the classifier on the training data
svm_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = svm_classifier.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Create a C-SVC classifier
svc_classifier = SVC()

# Train the classifier on the training data
svc_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = svc_classifier.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Create an SVC classifier with an RBF kernel
svc_classifier = SVC(kernel='rbf', C=1.0, gamma='scale')  # You can adjust C and gamma as needed

# Fit the classifier on the training data
svc_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = svc_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the results
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)

# Create an SVC classifier with a Polynomial kernel
svc_classifier = SVC(kernel='poly', degree=3, C=1.0)  # You can adjust the degree and C parameter as needed

# Fit the classifier on the training data
svc_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = svc_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the results
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)

# Define the parameter grid
param_grid = {
    'C': [1, 10, 50],
    'degree': [2, 3, 4]
}

# Create an SVC classifier with a Polynomial kernel
svc_classifier = SVC(kernel='poly')

# Create a GridSearchCV object
grid_search = GridSearchCV(svc_classifier, param_grid, cv=5, scoring='accuracy')

# Fit the GridSearchCV object on the training data
grid_search.fit(X_train, y_train)

# Get the best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Make predictions on the test data using the best model
y_pred = best_model.predict(X_test)

# Evaluate the best model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the results
print("Best Hyperparameters:", best_params)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)