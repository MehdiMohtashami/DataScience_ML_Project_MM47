# 1. Import Essential Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Models from Scikit-Learn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Preprocessing and Evaluation
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# To ignore warnings (optional)
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.style.use('fivethirtyeight')

# 2. Load the Dataset
df = pd.read_csv('car.csv') # Make sure the file is in your project directory

# 3. Initial Exploration (You've done some, let's do a bit more)
print("Dataset Shape:", df.shape)
print("\nFirst 10 entries:")
print(df.head(10))
print("\nDataset Info:")
print(df.info())
print("\nSummary:")
print(df.describe(include='all'))
print("\nClass Distribution:")
print(df['class'].value_counts())

# 4. Visual Exploratory Data Analysis (EDA)

# Let's see the distribution of the target variable 'class'
plt.figure(figsize=(10, 6))
order = df['class'].value_counts().index
sns.countplot(data=df, x='class', order=order, palette='viridis')
plt.title('Distribution of Car Acceptability Classes')
plt.savefig('Distribution_Car_AcceptabilityClasses.png')
plt.show()

# Now, let's see how each feature relates to the target 'class'
# We'll use factorplots (catplots in newer seaborn) for categorical features

features = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']

for feature in features:
    plt.figure(figsize=(12, 6))
    # Create a countplot for each feature, colored by the class
    sns.countplot(data=df, x=feature, hue='class', palette='viridis')
    plt.title(f'Distribution of Car Classes by {feature}')
    plt.legend(title='Car Class', loc='upper right')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# A pair plot can be interesting, but with many categories it might be crowded.
# Let's try a heatmap of correlations. First, we need to encode our data.
# Creating a temporary encoded dataframe for correlation analysis
df_encoded = df.copy()
le = LabelEncoder()
for col in df_encoded.columns:
    df_encoded[col] = le.fit_transform(df_encoded[col])

plt.figure(figsize=(10, 8))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Encoded Features')
plt.savefig('Correlation_Heatmap_EncodedFeatures.png')
plt.show()


# 5. Data Preprocessing

# Separate features (X) and target (y)
X = df.drop('class', axis=1)
y = df['class']

# Encoding the Target variable (y)
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)
# Let's see the mapping
print("Target Class Mapping:", dict(zip(le_target.classes_, le_target.transform(le_target.classes_))))

# For features (X), we have two main options: LabelEncoding or OneHotEncoding.
# Since there's no inherent ordinality in most features (e.g., 'med' is not necessarily
# the midpoint of 'low' and 'high'), OneHotEncoding is often safer.
# However, for tree-based models (like RandomForest), LabelEncoding can work just fine
# and is more memory efficient. Let's try LabelEncoding first for simplicity.

le_features = LabelEncoder()
X_encoded = X.apply(le_features.fit_transform)
# Let's see a sample of the encoded data
print("\nEncoded Features Sample:")
print(X_encoded.head())

# If you want to use models sensitive to arbitrary ordinality (like SVM, KNN),
# you should use OneHotEncoding later. We can do that in the pipeline for specific models.

# 6. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# 7. Initialize Multiple Classification Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'Support Vector Machine': SVC(),
    'Gradient Boosting': GradientBoostingClassifier()
}

# 8. Train, Predict and Evaluate Models
results = {}
for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    # Make predictions
    y_pred = model.predict(X_test)
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # Store the result
    results[name] = accuracy
    # Print a classification report for more details
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le_target.classes_))

# 9. Compare Model Accuracies
print("\n--- Model Accuracy Comparison ---")
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")

# Let's also visualize this comparison
plt.figure(figsize=(12, 6))
model_names = list(results.keys())
accuracies = list(results.values())
sns.barplot(x=accuracies, y=model_names, palette='viridis', orient='h')
plt.title('Model Accuracy Comparison')
plt.xlabel('Accuracy')
plt.xlim(0.7, 1.0) # Since we expect high accuracy, let's zoom in
plt.savefig('Model_Accuracy_Comparison.png')
plt.show()

# 10. More Robust Evaluation using Cross-Validation
print("\n--- Cross-Validation Scores (Mean) ---")
cv_results = {}
for name, model in models.items():
    # Perform 5-fold cross-validation
    cv_scores = cross_val_score(model, X_encoded, y_encoded, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_results[name] = cv_mean
    print(f"{name}: {cv_mean:.4f} (+/- {cv_scores.std() * 2:.4f})")

# Visualize CV results
plt.figure(figsize=(12, 6))
model_names = list(cv_results.keys())
cv_means = list(cv_results.values())
sns.barplot(x=cv_means, y=model_names, palette='viridis', orient='h')
plt.title('5-Fold Cross-Validation Mean Accuracy')
plt.xlabel('Mean Accuracy')
plt.xlim(0.7, 1.0)
plt.savefig('5-Fold_Cross-Validation_MeanAccuracy.png')
plt.show()
# 11. Hyperparameter Tuning for the Best Model (e.g., Random Forest)
# Defines the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the model
rf = RandomForestClassifier(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

# Fit GridSearchCV (this might take a while)
print("Starting Grid Search...")
grid_search.fit(X_train, y_train)

# Print the best parameters and the best score
print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")

# 12. Evaluate the Tuned Model on the Test Set
best_rf_model = grid_search.best_estimator_
y_pred_tuned = best_rf_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred_tuned)
print(f"\nTuned Random Forest Test Accuracy: {final_accuracy:.4f}")

# Detailed final report
print("\nTuned Model Classification Report:")
print(classification_report(y_test, y_pred_tuned, target_names=le_target.classes_))

# Plot a confusion matrix for the final model
cm = confusion_matrix(y_test, y_pred_tuned)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_target.classes_)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix - Tuned Random Forest')
plt.xticks(rotation=45)
plt.savefig('ConfusionMatrix-TunedRandomForest.png')
plt.show()

# 13. Save the Final Model and Encoders for Future Use
# import joblib
#
# # Create a dictionary of artifacts to save
# model_artifacts = {
#     'model': best_rf_model,
#     'target_encoder': le_target,
#     'feature_encoder': le_features,
#     'feature_names': list(X.columns),
#     'target_names': list(le_target.classes_)
# }
#
# # Save the artifacts to a file
# joblib.dump(model_artifacts, 'car_acceptability_classifier.pkl')
#
# print("Model and encoders saved successfully as 'car_acceptability_classifier.pkl'")

# 14. Final Model Saving with Joblib

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Load your dataset (for re-training the final model on the entire dataset)
df = pd.read_csv('car.csv')

# 2. Separate features and target
X = df.drop('class', axis=1)
y = df['class']

# 3. Encode the features (Using LabelEncoding as before)
le_features = LabelEncoder()
# We need to fit the encoder on each column. Using .apply to do it for all feature columns.
X_encoded = X.apply(le_features.fit_transform)

# 4. Encode the target variable
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

# 5. (Optional but Recommended) Re-train the Tuned model on the ENTIRE dataset
# The best parameters we found from GridSearchCV
best_params = {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}

# Create the final model with the best parameters
final_model = RandomForestClassifier(**best_params, random_state=42)
# Train it on all our data
final_model.fit(X_encoded, y_encoded)

# 6. Create a dictionary of all artifacts needed for future predictions
model_artifacts = {
    'model': final_model,
    'feature_encoder': le_features,  # The fitted LabelEncoder for features
    'target_encoder': le_target,     # The fitted LabelEncoder for the target
    'feature_names': list(X.columns), # List of feature names
    'target_names': list(le_target.classes_) # List of target class names
}

# 7. Save the artifacts to a file using joblib
joblib.dump(model_artifacts, 'car_acceptability_tuned_rf_model.pkl')

print("✅ Model and encoders saved successfully as 'car_acceptability_tuned_rf_model.pkl'")
print(f"   Model Type: Tuned Random Forest")
print(f"   Model Parameters: {best_params}")
print(f"   Features: {model_artifacts['feature_names']}")
print(f"   Target Classes: {list(model_artifacts['target_encoder'].classes_)}")