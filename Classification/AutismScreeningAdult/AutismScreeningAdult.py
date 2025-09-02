# Comprehensive data review and resolution of potential Data Leakage problems
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, \
    f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import joblib
import warnings

warnings.filterwarnings('ignore')

# Load the main dataset
df = pd.read_csv('Autism-Adult-Data.csv')

print("=" * 60)
print("Initial data review and identification of suspicious features")
print("=" * 60)

# Examine the relationship between result and Class/ASD
print("Distribution of result based on Class/ASD:")
result_asd_crosstab = pd.crosstab(df['result'], df['Class/ASD'], margins=True)
print(result_asd_crosstab)

# Checking the correlation between result and other numerical features
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'result' in numeric_cols and 'Class/ASD' in df.columns:
    # Convert Class/ASD to a number to calculate correlation
    df_corr = df.copy()
    le = LabelEncoder()
    df_corr['Class/ASD_num'] = le.fit_transform(df_corr['Class/ASD'])

    # Correlation calculation
    correlation = df_corr['result'].corr(df_corr['Class/ASD_num'])
    print(f"\nCorrelation between result and Class/ASD: {correlation:.4f}")

# Data preprocessing
print("\n" + "=" * 60)
print("Data preprocessing")
print("=" * 60)

# Handling missing values
df['age'].fillna(df['age'].median(), inplace=True)
df['ethnicity'].fillna('Unknown', inplace=True)
df['relation'].fillna('Unknown', inplace=True)

# Remove unnecessary columns
df.drop('age_desc', axis=1, inplace=True)

# Convert categorical variables to numerical
binary_cols = ['gender', 'jundice', 'austim', 'used_app_before', 'Class/ASD']
le = LabelEncoder()
for col in binary_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col])

# One-Hot Encoding for categorical variables
categorical_cols = ['ethnicity', 'contry_of_res', 'relation']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print(f"Data dimensions after preprocessing: {df.shape}")

# Two different scenarios for modeling
print("\n" + "=" * 60)
print("Comparison of two scenarios: with and without the result feature")
print("=" * 60)

scenarios = {
    'With result feature': df.drop('Class/ASD', axis=1),
    'Without result feature': df.drop(['Class/ASD', 'result'], axis=1)
}

results_comparison = {}

for scenario_name, X in scenarios.items():
    print(f"\n{'=' * 30} {scenario_name} {'=' * 30}")

    y = df['Class/ASD']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Machine": SVC(random_state=42, class_weight='balanced', probability=True),
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=5),
        "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100,
                                                max_depth=5),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=3),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss', max_depth=3)
    }

    scenario_results = {}

    for name, model in models.items():
        # Train the model
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model.predict(X_test_scaled)

        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        scenario_results[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1
        }

        # Check for overfitting
        y_train_pred = model.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, y_train_pred)
        acc_diff = train_acc - acc

        print(f"{name}:")
        print(f"  Test Accuracy: {acc:.4f}, Train Accuracy: {train_acc:.4f}, Difference: {acc_diff:.4f}")
        if acc_diff > 0.05:
            print("  ⚠️ Potential overfitting")

    # Store results for this scenario
    results_comparison[scenario_name] = scenario_results

# Compare results of the two scenarios
print("\n" + "=" * 60)
print("Comprehensive comparison of model performance in two scenarios")
print("=" * 60)

for model_name in models.keys():
    print(f"\n{model_name}:")
    print("-" * len(model_name))

    for scenario_name in scenarios.keys():
        if model_name in results_comparison[scenario_name]:
            results = results_comparison[scenario_name][model_name]
            print(f"  {scenario_name}:")
            print(f"    Accuracy: {results['accuracy']:.4f}, F1-score: {results['f1_score']:.4f}")

# Select the best scenario and model
print("\n" + "=" * 60)
print("Selecting the best model based on F1-score")
print("=" * 60)

best_scenario = None
best_model_name = None
best_f1 = 0

for scenario_name, scenario_results in results_comparison.items():
    for model_name, results in scenario_results.items():
        if results['f1_score'] > best_f1:
            best_f1 = results['f1_score']
            best_model_name = model_name
            best_scenario = scenario_name

print(f"Best model: {best_model_name}")
print(f"Best scenario: {best_scenario}")
print(f"Best F1-score: {best_f1:.4f}")

# Train and save the best model
print("\n" + "=" * 60)
print("Training and saving the best model")
print("=" * 60)

# Select appropriate data based on the best scenario
if best_scenario == 'With result feature':
    X_final = df.drop('Class/ASD', axis=1)
else:
    X_final = df.drop(['Class/ASD', 'result'], axis=1)

y_final = df['Class/ASD']

# Split the data
X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
    X_final, y_final, test_size=0.2, random_state=42, stratify=y_final
)

# Standardization
scaler_final = StandardScaler()
X_train_final_scaled = scaler_final.fit_transform(X_train_final)
X_test_final_scaled = scaler_final.transform(X_test_final)

# Train the best model
if best_model_name == "Logistic Regression":
    best_model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
elif best_model_name == "K-Nearest Neighbors":
    best_model = KNeighborsClassifier()
elif best_model_name == "Support Vector Machine":
    best_model = SVC(random_state=42, class_weight='balanced', probability=True)
elif best_model_name == "Decision Tree":
    best_model = DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=5)
elif best_model_name == "Random Forest":
    best_model = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100, max_depth=5)
elif best_model_name == "Gradient Boosting":
    best_model = GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=3)
elif best_model_name == "XGBoost":
    best_model = XGBClassifier(random_state=42, eval_metric='logloss', max_depth=3)

best_model.fit(X_train_final_scaled, y_train_final)

# Final evaluation
y_pred_final = best_model.predict(X_test_final_scaled)
final_acc = accuracy_score(y_test_final, y_pred_final)
final_f1 = f1_score(y_test_final, y_pred_final)

print(f"Final accuracy of the best model: {final_acc:.4f}")
print(f"Final F1-score: {final_f1:.4f}")

# Save the model and scaler
joblib.dump(best_model, 'best_model_final.joblib')
joblib.dump(scaler_final, 'scaler_final.joblib')

# Save feature names
feature_names = list(X_final.columns)
joblib.dump(feature_names, 'feature_names_final.joblib')

print("\nModel, scaler, and feature names successfully saved.")

# Analyze feature importance (if the model is tree-based)
if hasattr(best_model, 'feature_importances_'):
    print("\n" + "=" * 60)
    print("Feature importance analysis in the best model")
    print("=" * 60)

    feature_importance = pd.DataFrame({
        'feature': X_final.columns,
        'importance': best_model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title(f'Important Features - {best_model_name}')
    plt.tight_layout()
    plt.show()

    print("Top ten important features:")
    print(feature_importance[['feature', 'importance']].to_string(index=False))

print("\n" + "=" * 60)
print("Final conclusion")
print("=" * 60)

if best_scenario == 'With result feature' and best_f1 > 0.95:
    print("⚠️ Warning: The best model uses the result feature, which may cause data leakage!")
    print("Recommendation: In a production environment, use the scenario without the result feature.")
elif best_f1 > 0.85:
    print("✅ The selected model performs well and is suitable for production use.")
else:
    print("🔍 The model needs improvement. Consider the following techniques:")
    print("   - Hyperparameter tuning")
    print("   - Using techniques to address imbalanced data")
    print("   - Increasing training data")

print(f"\nBest final model: {best_model_name}")
print(f"Selected scenario: {best_scenario}")

# Modified code for using the real model
# Complete code for training and saving the real model without Data Leakage
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Chart settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
sns.set_palette("husl")

# Load the main dataset for testing
df = pd.read_csv('Autism-Adult-Data.csv')

# Preprocess data (same as before)
df['age'].fillna(df['age'].median(), inplace=True)
df['ethnicity'].fillna('Unknown', inplace=True)
df['relation'].fillna('Unknown', inplace=True)
df.drop('age_desc', axis=1, inplace=True)

from sklearn.preprocessing import LabelEncoder

binary_cols = ['gender', 'jundice', 'austim', 'used_app_before', 'Class/ASD']
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

categorical_cols = ['ethnicity', 'contry_of_res', 'relation']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Prepare data for testing
X_with_result = df.drop('Class/ASD', axis=1)
X_without_result = df.drop(['Class/ASD', 'result'], axis=1)
y = df['Class/ASD']

# Split the data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_without_result, y, test_size=0.2, random_state=42, stratify=y)

# Standardization
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)

print("=" * 60)
print("Comprehensive comparison of all saved models")
print("=" * 60)

# List of all saved models
model_files = {
    'final_model': 'best_model_final.joblib',
    'real_model': 'best_model_real.joblib',
    'logistic_regression': 'models/logistic_regression_model.joblib',
    'random_forest': 'models/random_forest_model.joblib',
    'gradient_boosting': 'models/gradient_boosting_model.joblib',
    'xgboost': 'models/xgboost_model.joblib'
}

results = []

# Test each model
for model_name, file_path in model_files.items():
    try:
        # Load the model
        model = joblib.load(file_path)

        # Predict
        if 'result' in model_name:
            # For models trained with result, we need different data
            X_test_special = X_with_result.iloc[X_test.index]
            X_test_special_scaled = scaler.transform(X_test_special)
            y_pred = model.predict(X_test_special_scaled)
        else:
            # For models without result
            y_pred = model.predict(X_test_scaled)

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append({
            'model': model_name,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1
        })

        print(f"{model_name}: Accuracy = {acc:.4f}, F1-score = {f1:.4f}")

    except FileNotFoundError:
        print(f"File {file_path} not found")
    except Exception as e:
        print(f"Error loading {model_name}: {str(e)}")

# Create DataFrame from results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('f1_score', ascending=False)

print("\n" + "=" * 60)
print("Model ranking based on F1-Score")
print("=" * 60)
print(results_df[['model', 'accuracy', 'f1_score']].round(4))

# Plot comparison chart
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Comprehensive comparison of different model performances', fontsize=16, fontweight='bold')

# Accuracy chart
axes[0, 0].barh(results_df['model'], results_df['accuracy'], color='skyblue')
axes[0, 0].set_title('Model Accuracy')
axes[0, 0].set_xlabel('Accuracy')
axes[0, 0].set_xlim(0, 1)

# F1-Score chart
axes[0, 1].barh(results_df['model'], results_df['f1_score'], color='lightgreen')
axes[0, 1].set_title('Model F1-Score')
axes[0, 1].set_xlabel('F1-Score')
axes[0, 1].set_xlim(0, 1)

# Precision chart
axes[1, 0].barh(results_df['model'], results_df['precision'], color='lightcoral')
axes[1, 0].set_title('Model Precision')
axes[1, 0].set_xlabel('Precision')
axes[1, 0].set_xlim(0, 1)

# Recall chart
axes[1, 1].barh(results_df['model'], results_df['recall'], color='gold')
axes[1, 1].set_title('Model Recall')
axes[1, 1].set_xlabel('Recall')
axes[1, 1].set_xlim(0, 1)

plt.tight_layout()
plt.show()

# Radar chart for comparing all metrics
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, polar=True)

# Calculate angles
categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

# Plot for each model
colors = plt.cm.viridis(np.linspace(0, 1, len(results_df)))

for idx, (_, row) in enumerate(results_df.iterrows()):
    values = [
        row['accuracy'],
        row['precision'],
        row['recall'],
        row['f1_score']
    ]
    values += values[:1]

    ax.plot(angles, values, color=colors[idx], linewidth=2, label=row['model'])
    ax.fill(angles, values, color=colors[idx], alpha=0.1)

# Chart settings
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_yticklabels([])
ax.set_title('Comprehensive model comparison with Radar chart', size=15, y=1.1)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.tight_layout()
plt.show()

# Find the best model
best_model_row = results_df.iloc[0]
best_model_name = best_model_row['model']
best_model_file = model_files[best_model_name]

print("\n" + "=" * 60)
print("Final conclusion and selection of the best model")
print("=" * 60)
print(f"Best model: {best_model_name}")
print(f"Accuracy: {best_model_row['accuracy']:.4f}")
print(f"F1-Score: {best_model_row['f1_score']:.4f}")
print(f"Precision: {best_model_row['precision']:.4f}")
print(f"Recall: {best_model_row['recall']:.4f}")

# Warning about models with data leakage
models_with_result = [name for name in results_df['model'] if 'result' in name]
if models_with_result:
    print(f"\n⚠️ Warning: The following models may have data leakage:")
    for model in models_with_result:
        print(f"   - {model}")

print("\n✅ Recommendation: Use best_model_real.joblib")
print("   because it was trained without data leakage and has realistic performance")

# Display feature importance for the best model
try:
    best_model = joblib.load(best_model_file)
    if hasattr(best_model, 'feature_importances_'):
        print("\n" + "=" * 60)
        print("Top ten important features in the best model")
        print("=" * 60)

        feature_importance = pd.DataFrame({
            'feature': X_without_result.columns,
            'importance': best_model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)

        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Top ten important features in the best model')
        plt.tight_layout()
        plt.show()

        print(feature_importance[['feature', 'importance']].to_string(index=False))

except Exception as e:
    print(f"Error displaying feature importance: {str(e)}")