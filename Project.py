# Name: Tommy-Nam Nguyen

# Project Overview: 
# Goal: To predict the success of movies based on key features such as budget, genre, runtime, ratings, and popularity.
# Outcome: Provide insights into which factors contribute most to movie success, aiding decision-making in the film industry.

import kagglehub
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, accuracy_score
from sklearn.model_selection import train_test_split

# Download latest version
path = kagglehub.dataset_download("omkarborikar/top-10000-popular-movies")

print("Path to dataset files:", path)

dataset_path = r"C:\Users\TheTr\.cache\kagglehub\datasets\omkarborikar\top-10000-popular-movies\versions\5\Top_10000_Movies.csv"

# ---------------------------------------------------------------------------------- #
# Visualization #
# ---------------------------------------------------------------------------------- #

# >> Read file in chunks <<
chunk_size = 10_000  # Adjust chunk size as needed
chunks = []

try:
    for chunk in pd.read_csv(dataset_path, chunksize=chunk_size, engine='python'):
        chunks.append(chunk)
    data = pd.concat(chunks, ignore_index=True)  # Combine all chunks
    print("File loaded successfully!")
    print(data.head())
except Exception as e:
    print("Error loading file:", e)
print(data.info())

# >> Revenue Distribution <<
sns.histplot(data['revenue'], kde=True, bins=30)
plt.title('Revenue Distribution')
plt.xlabel('Revenue')
plt.ylabel('Frequency')
plt.show()

# >> Revenue by Genre <<
# Extract primary genre
data['primary_genre'] = data['genre'].apply(lambda x: x.split('|')[0] if isinstance(x, str) else x)
# Filter top genres by occurrence
top_genres = data['primary_genre'].value_counts().head(10).index
filtered_data = data[data['primary_genre'].isin(top_genres)]
# Plot with cleaned data
sns.boxplot(x='primary_genre', y='revenue', data=filtered_data)
plt.title('Revenue by Primary Genre')
plt.xticks(rotation=15)
plt.xlabel('Primary Genre')
plt.ylabel('Revenue')
plt.show()

# >> Correlation Heatmap <<
# Drop irrelevant columns like 'id'
numeric_data = data.drop(columns=['id']).select_dtypes(include=['float64', 'int64'])
# Handle missing values in numeric data
numeric_data = numeric_data.fillna(0)  # Replace NaNs with 0 (you can use other strategies like mean or median)
# Compute correlation matrix
corr = numeric_data.corr()
# Plot correlation heatmap
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.xticks(rotation=15)
plt.show()

print(data.describe())
print(data.isnull().sum())

# ---------------------------------------------------------------------------------- #
# Preprocessing #
# ---------------------------------------------------------------------------------- #

# Handle Missing Values
print("\nHandling missing values...")

# Fill numerical columns with median
data['revenue'] = data['revenue'].fillna(data['revenue'].median())
data['runtime'] = data['runtime'].fillna(data['runtime'].median())

# Drop rows with critical missing values
data.dropna(subset=['popularity'], inplace=True)

# Drop Irrelevant Columns
print("\nDropping irrelevant columns...")
data.drop(columns=['Unnamed: 0', 'original_title', 'tagline', 'overview', 'release_date'], inplace=True)

# Ensure genres are split correctly
# If "genre" is a stringified list, extract the first genre
data['primary_genre'] = data['genre'].apply(
    lambda x: x.split('|')[0] if isinstance(x, str) else x
)

# Encode Categorical Data
print("\nEncoding categorical variables...")
# One-hot encode genre and language
data = pd.get_dummies(data, columns=['primary_genre', 'original_language'], drop_first=True)

# Drop any remaining non-numeric columns
non_numeric_columns = data.select_dtypes(include=['object', 'bool']).columns
if not non_numeric_columns.empty:
    print(f"Dropping non-numeric columns: {non_numeric_columns}")
    data.drop(columns=non_numeric_columns, inplace=True)

# Scale Numerical Features
print("\nScaling numerical features...")
scaler = StandardScaler()
numerical_features = ['popularity', 'runtime', 'vote_average', 'vote_count']

# Apply scaling to numerical features
data[numerical_features] = scaler.fit_transform(data[numerical_features])

print("\nData Preprocessing Complete!")

print(data.describe())
print(data.isnull().sum())
print(data.info())  # Ensure no missing values in critical columns
print(data.head())  # Inspect the first few rows

# ---------------------------------------------------------------------------------- #
# ML Algorithms #
# ---------------------------------------------------------------------------------- #

# Define the target variable (is_success) as movies with revenue above median
data['is_success'] = (data['revenue'] > data['revenue'].median()).astype(int)

# Drop irrelevant columns before feature splitting
X = data.drop(columns=['id', 'revenue', 'is_success'])
y = data['is_success']

# Ensure all data in X is numeric
assert X.select_dtypes(include=['object']).empty, "Non-numeric data remains in X."

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ############## Logistic Regression ##############
lr_model = LogisticRegression(max_iter=500)
lr_model.fit(X_train, y_train)

print("Logistic Regression Accuracy:", lr_model.score(X_test, y_test))

# ############## Random Forest ##############
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

print("Random Forest Accuracy:", rf_model.score(X_test, y_test))

# Classification Reports
# Logistic Regression
y_pred_lr = lr_model.predict(X_test)
print("Logistic Regression Report:")
print(classification_report(y_test, y_pred_lr))

# Random Forest
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Report:")
print(classification_report(y_test, y_pred_rf))


# ############## Naive Bayes model ##############
nb_model = GaussianNB()

# Train the model
nb_model.fit(X_train, y_train)

# Predictions on the test set
y_pred_nb = nb_model.predict(X_test)

# Evaluate the model
nb_accuracy = accuracy_score(y_test, y_pred_nb)
print("Naive Bayes Accuracy:", nb_accuracy)

# Classification report
print("Naive Bayes Classification Report:")
print(classification_report(y_test, y_pred_nb))

# Confusion Matrix
nb_cm = confusion_matrix(y_test, y_pred_nb)
print("Naive Bayes Confusion Matrix:\n", nb_cm)

# Plot Confusion Matrix
sns.heatmap(nb_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Successful', 'Successful'], yticklabels=['Not Successful', 'Successful'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Naive Bayes Confusion Matrix')
plt.show()

# ---------------------------------------------------------------------------------- #
# Observations & Analysis #
# ---------------------------------------------------------------------------------- #

# AUC-ROC Comparison
lr_probs = lr_model.predict_proba(X_test)[:, 1]
rf_probs = rf_model.predict_proba(X_test)[:, 1]
nb_probs = nb_model.predict_proba(X_test)[:, 1]
lr_auc = roc_auc_score(y_test, lr_probs)
rf_auc = roc_auc_score(y_test, rf_probs)
nb_auc = roc_auc_score(y_test, nb_probs)

print(f"Logistic Regression AUC: {lr_auc:.2f}")
print(f"Random Forest AUC: {rf_auc:.2f}")
print(f"Naive Bayes AUC: {nb_auc:.2f}")

# ROC Curve
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
nb_fpr, nb_tpr, _ = roc_curve(y_test, nb_probs)

plt.figure(figsize=(10, 6))
plt.plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC = {lr_auc:.2f})')
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.2f})')
plt.plot(nb_fpr, nb_tpr, label=f'Naive Bayes (AUC = {nb_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random chance line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.show()

# Feature Importance
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
important_features = feature_importances.nlargest(10)

# Plot top features
important_features.plot(kind='barh', figsize=(10, 6))
plt.title('Top 10 Features Contributing to Movie Success')
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.show()