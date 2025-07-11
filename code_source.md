Extracted 'source' values:

# Concrete Compressive Strength Prediction - Machine Learning Analysis


## Overview
This notebook analyzes concrete compressive strength under different exposure conditions (NaCl, Water, Na2SO4) using various machine learning models. The analysis includes model comparison, evaluation metrics, and comprehensive visualizations.


## 1. Import Required Librariesimport pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")
## 2. Data Loading and Preprocessing# Load datasets
nacl_data = pd.read_csv('NaCl datasets.csv')
water_data = pd.read_csv('water datasets.csv')
na2so4_data = pd.read_csv('Na2So4 datasets.csv')

# Add exposure type column
nacl_data['exposure_type'] = 'NaCl'
water_data['exposure_type'] = 'Water'
na2so4_data['exposure_type'] = 'Na2SO4'

# Combine all datasets
combined_data = pd.concat([nacl_data, water_data, na2so4_data], ignore_index=True)

print("Dataset Information:")
print(f"Combined dataset shape: {combined_data.shape}")
print(f"NaCl dataset shape: {nacl_data.shape}")
print(f"Water dataset shape: {water_data.shape}")
print(f"Na2SO4 dataset shape: {na2so4_data.shape}")# Display basic statistics
print("\nDataset Info:")
print(combined_data.info())print("\nBasic Statistics:")
print(combined_data.describe())
## 3. Data Exploration and Visualization# Create comprehensive visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Compressive strength vs exposure days for different conditions
for i, (data, label) in enumerate([(nacl_data, 'NaCl'), (water_data, 'Water'), (na2so4_data, 'Na2SO4')]):
    axes[0, i].plot(data['No_of_days_of_NaCl_Exposure'], data['compressive_stregth'], 
                    marker='o', linewidth=2, markersize=6)
    axes[0, i].set_title(f'Compressive Strength vs Days - {label}', fontsize=12, fontweight='bold')
    axes[0, i].set_xlabel('Days of Exposure')
    axes[0, i].set_ylabel('Compressive Strength (MPa)')
    axes[0, i].grid(True, alpha=0.3)

# 2. Distribution of compressive strength by exposure type
axes[1, 0].boxplot([nacl_data['compressive_stregth'], water_data['compressive_stregth'], 
                    na2so4_data['compressive_stregth']], 
                   labels=['NaCl', 'Water', 'Na2SO4'])
axes[1, 0].set_title('Compressive Strength Distribution by Exposure Type', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Compressive Strength (MPa)')

# 3. Correlation heatmap
numeric_cols = combined_data.select_dtypes(include=[np.number]).columns
correlation_matrix = combined_data[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            ax=axes[1, 1], fmt='.2f')
axes[1, 1].set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold')

# 4. Scatter plot: Days vs Compressive Strength (all conditions)
colors = ['red', 'blue', 'green']
for i, (data, label, color) in enumerate([(nacl_data, 'NaCl', 'red'), 
                                          (water_data, 'Water', 'blue'), 
                                          (na2so4_data, 'Na2SO4', 'green')]):
    axes[1, 2].scatter(data['No_of_days_of_NaCl_Exposure'], data['compressive_stregth'], 
                       c=color, label=label, alpha=0.7, s=50)
axes[1, 2].set_title('Compressive Strength vs Days (All Conditions)', fontsize=12, fontweight='bold')
axes[1, 2].set_xlabel('Days of Exposure')
axes[1, 2].set_ylabel('Compressive Strength (MPa)')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
## 4. Feature Engineering and Data Preparation# Feature engineering
def prepare_data(df):
    # Create new features
    df = df.copy()
    df['w_b_ratio'] = df['w_b']
    df['total_scm'] = df['scm_flyash'] + df['scm_ggbs']
    df['fiber_aspect_ratio'] = df['fiber_length'] / df['fiber_diameter']
    df['fiber_volume_fraction'] = df['perc_of_fibre'] / 100
    df['days_squared'] = df['No_of_days_of_NaCl_Exposure'] ** 2
    df['days_log'] = np.log1p(df['No_of_days_of_NaCl_Exposure'])
    
    return df

# Prepare datasets
nacl_prepared = prepare_data(nacl_data)
water_prepared = prepare_data(water_data)
na2so4_prepared = prepare_data(na2so4_data)
combined_prepared = prepare_data(combined_data)

# Encode exposure type for combined dataset
le = LabelEncoder()
combined_prepared['exposure_type_encoded'] = le.fit_transform(combined_prepared['exposure_type'])

print("Feature Engineering Complete!")
print(f"New features added: {list(set(combined_prepared.columns) - set(combined_data.columns))}")
## 5. Model Definition and Training Function# Define all models
def get_models():
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=1.0),
        'Elastic Net': ElasticNet(alpha=1.0, l1_ratio=0.5),
        'Bayesian Ridge': BayesianRidge(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
        'CatBoost': CatBoostRegressor(iterations=100, random_state=42, verbose=False),
        'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
        'KNN': KNeighborsRegressor(n_neighbors=5),
        'Gaussian Process': GaussianProcessRegressor(kernel=C(1.0) * RBF(1.0), random_state=42)
    }
    return models

# Training and evaluation function
def train_and_evaluate_models(X_train, X_test, y_train, y_test, dataset_name):
    models = get_models()
    results = {}
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n{'='*60}")
    print(f"Training Models for {dataset_name} Dataset")
    print(f"{'='*60}")
    
    for name, model in models.items():
        try:
            # Train model
            if name in ['SVR', 'KNN', 'Gaussian Process']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            if name in ['SVR', 'KNN', 'Gaussian Process']:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            results[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'CV_R2_mean': cv_scores.mean(),
                'CV_R2_std': cv_scores.std(),
                'Model': model,
                'Predictions': y_pred
            }
            
            print(f"{name:15} | R² = {r2:.4f} | RMSE = {rmse:.4f} | MAE = {mae:.4f} | CV R² = {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
            
        except Exception as e:
            print(f"{name:15} | Error: {str(e)}")
            continue
    
    return results, scaler

# Feature selection function
def select_features(df, target_col, include_exposure_type=False):
    # Always exclude the target column and the original string exposure_type column
    exclude_cols = [target_col, 'exposure_type']
    
    # If we don't want to include exposure type at all, also exclude the encoded version
    if not include_exposure_type:
        exclude_cols.append('exposure_type_encoded')
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols
    ## 6. Model Training and Evaluation for Each Dataset# Results storage
all_results = {}

# 1. NaCl Dataset
print("Processing NaCl Dataset...")
feature_cols = select_features(nacl_prepared, 'compressive_stregth')
X_nacl = nacl_prepared[feature_cols]
y_nacl = nacl_prepared['compressive_stregth']

X_train_nacl, X_test_nacl, y_train_nacl, y_test_nacl = train_test_split(
    X_nacl, y_nacl, test_size=0.3, random_state=42
)

nacl_results, nacl_scaler = train_and_evaluate_models(
    X_train_nacl, X_test_nacl, y_train_nacl, y_test_nacl, "NaCl"
)
all_results['NaCl'] = nacl_results# 2. Water Dataset
print("\nProcessing Water Dataset...")
X_water = water_prepared[feature_cols]
y_water = water_prepared['compressive_stregth']

X_train_water, X_test_water, y_train_water, y_test_water = train_test_split(
    X_water, y_water, test_size=0.3, random_state=42
)

water_results, water_scaler = train_and_evaluate_models(
    X_train_water, X_test_water, y_train_water, y_test_water, "Water"
)
all_results['Water'] = water_results# 3. Na2SO4 Dataset
print("\nProcessing Na2SO4 Dataset...")
X_na2so4 = na2so4_prepared[feature_cols]
y_na2so4 = na2so4_prepared['compressive_stregth']

X_train_na2so4, X_test_na2so4, y_train_na2so4, y_test_na2so4 = train_test_split(
    X_na2so4, y_na2so4, test_size=0.3, random_state=42
)

na2so4_results, na2so4_scaler = train_and_evaluate_models(
    X_train_na2so4, X_test_na2so4, y_train_na2so4, y_test_na2so4, "Na2SO4"
)
all_results['Na2SO4'] = na2so4_results
# 4. Combined Dataset
print("\nProcessing Combined Dataset...")
combined_feature_cols = select_features(combined_prepared, 'compressive_stregth', include_exposure_type=True)
X_combined = combined_prepared[combined_feature_cols]
y_combined = combined_prepared['compressive_stregth']

X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(
    X_combined, y_combined, test_size=0.3, random_state=42
)

combined_results, combined_scaler = train_and_evaluate_models(
    X_train_combined, X_test_combined, y_train_combined, y_test_combined, "Combined"
)
all_results['Combined'] = combined_results
## 7. Results Analysis and Best Model Selection# Create results summary
def create_results_summary(all_results):
    summary_data = []
    
    for dataset_name, results in all_results.items():
        for model_name, metrics in results.items():
            summary_data.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'R²': metrics['R2'],
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'CV_R²_mean': metrics['CV_R2_mean'],
                'CV_R²_std': metrics['CV_R2_std']
            })
    
    return pd.DataFrame(summary_data)

# Generate summary
results_summary = create_results_summary(all_results)

# Find best models for each dataset
best_models = {}
for dataset in ['NaCl', 'Water', 'Na2SO4', 'Combined']:
    dataset_results = results_summary[results_summary['Dataset'] == dataset]
    best_model = dataset_results.loc[dataset_results['R²'].idxmax()]
    best_models[dataset] = best_model

print("\n" + "="*80)
print("BEST MODEL FOR EACH DATASET")
print("="*80)

for dataset, best_model in best_models.items():
    print(f"\n{dataset} Dataset:")
    print(f"  Best Model: {best_model['Model']}")
    print(f"  R² Score: {best_model['R²']:.4f}")
    print(f"  RMSE: {best_model['RMSE']:.4f}")
    print(f"  MAE: {best_model['MAE']:.4f}")
    print(f"  CV R² (mean±std): {best_model['CV_R²_mean']:.4f}±{best_model['CV_R²_std']:.4f}")# Display top 5 models for each dataset
print("\n" + "="*80)
print("TOP 5 MODELS FOR EACH DATASET")
print("="*80)

for dataset in ['NaCl', 'Water', 'Na2SO4', 'Combined']:
    print(f"\n{dataset} Dataset - Top 5 Models:")
    dataset_results = results_summary[results_summary['Dataset'] == dataset]
    top_5 = dataset_results.nlargest(5, 'R²')[['Model', 'R²', 'RMSE', 'MAE', 'CV_R²_mean']]
    print(top_5.to_string(index=False))
    ## 8. Comprehensive Visualizations# Create comprehensive visualization plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # Changed to 2x3 grid

# 1. Model Performance Comparison by Dataset (first 4 subplots)
datasets = ['NaCl', 'Water', 'Na2SO4', 'Combined']
colors = ['red', 'blue', 'green', 'purple']

for i, dataset in enumerate(datasets):
    dataset_results = results_summary[results_summary['Dataset'] == dataset]
    top_10 = dataset_results.nlargest(10, 'R²')
    
    # Map to 2x3 grid positions
    if i < 2:
        ax = axes[0, i]  # First row: positions 0,0 and 0,1
    else:
        ax = axes[1, i-2]  # Second row: positions 1,0 and 1,1
    
    bars = ax.bar(range(len(top_10)), top_10['R²'], color=colors[i], alpha=0.7)
    ax.set_title(f'Top 10 Models - {dataset} Dataset', fontsize=12, fontweight='bold')
    ax.set_xlabel('Models')
    ax.set_ylabel('R² Score')
    ax.set_xticks(range(len(top_10)))
    ax.set_xticklabels(top_10['Model'], rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, top_10['R²']):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)

# 2. RMSE Comparison (position 0,2)
ax = axes[0, 2]
for i, dataset in enumerate(datasets):
    dataset_results = results_summary[results_summary['Dataset'] == dataset]
    top_5 = dataset_results.nlargest(5, 'R²')
    ax.plot(top_5['Model'], top_5['RMSE'], marker='o', label=dataset, linewidth=2, markersize=6)

ax.set_title('RMSE Comparison - Top 5 Models per Dataset', fontsize=12, fontweight='bold')
ax.set_xlabel('Models')
ax.set_ylabel('RMSE')
ax.legend()
ax.grid(True, alpha=0.3)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

# 3. Overall Best Models Comparison (position 1,2)
ax = axes[1, 2]
best_models_df = pd.DataFrame(best_models).T
metrics = ['R²', 'RMSE', 'MAE']
x = np.arange(len(datasets))
width = 0.25

for i, metric in enumerate(metrics):
    values = [best_models[dataset][metric] for dataset in datasets]
    ax.bar(x + i*width, values, width, label=metric, alpha=0.8)

ax.set_title('Best Model Performance by Dataset', fontsize=12, fontweight='bold')
ax.set_xlabel('Dataset')
ax.set_ylabel('Score')
ax.set_xticks(x + width)
ax.set_xticklabels(datasets)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Separate plot for Cross-validation scores
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

cv_data = []
cv_labels = []
for dataset in datasets:
    dataset_results = results_summary[results_summary['Dataset'] == dataset]
    best_model_name = best_models[dataset]['Model']
    best_cv_scores = dataset_results[dataset_results['Model'] == best_model_name]['CV_R²_mean'].values[0]
    cv_data.append(best_cv_scores)
    cv_labels.append(f"{dataset}\n{best_model_name}")

bars = ax.bar(range(len(cv_data)), cv_data, color=colors, alpha=0.7)
ax.set_title('Cross-Validation R² Scores - Best Models', fontsize=12, fontweight='bold')
ax.set_xlabel('Dataset (Best Model)')
ax.set_ylabel('CV R² Score')
ax.set_xticks(range(len(cv_labels)))
ax.set_xticklabels(cv_labels, rotation=45, ha='right')
ax.grid(True, alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, cv_data):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
            f'{value:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()
## 9. Prediction vs Actual Plots# Create prediction vs actual plots for best models
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

test_data = [
    (y_test_nacl, 'NaCl'),
    (y_test_water, 'Water'), 
    (y_test_na2so4, 'Na2SO4'),
    (y_test_combined, 'Combined')
]

for i, (y_test, dataset_name) in enumerate(test_data):
    ax = axes[i//2, i%2]
    
    # Get best model predictions
    best_model_name = best_models[dataset_name]['Model']
    y_pred = all_results[dataset_name][best_model_name]['Predictions']
    r2 = all_results[dataset_name][best_model_name]['R2']
    
    # Plot
    ax.scatter(y_test, y_pred, alpha=0.6, s=50)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Actual Compressive Strength (MPa)')
    ax.set_ylabel('Predicted Compressive Strength (MPa)')
    ax.set_title(f'{dataset_name} - {best_model_name}\nR² = {r2:.4f}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add perfect prediction line
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', alpha=0.5, label='Perfect Prediction')

plt.tight_layout()
plt.show()
## 10. Feature Importance Analysis# Feature importance for tree-based models
def plot_feature_importance(model, feature_names, dataset_name, ax):
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        ax.bar(range(len(importance)), importance[indices], alpha=0.7)
        ax.set_title(f'Feature Importance - {dataset_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Features')
        ax.set_ylabel('Importance')
        ax.set_xticks(range(len(importance)))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
        ax.grid(True, alpha=0.3)

# Plot feature importance for tree-based best models
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

feature_sets = [
    (feature_cols, 'NaCl'),
    (feature_cols, 'Water'),
    (feature_cols, 'Na2SO4'),
    (combined_feature_cols, 'Combined')
]

for i, (features, dataset_name) in enumerate(feature_sets):
    ax = axes[i//2, i%2]
    
    best_model_name = best_models[dataset_name]['Model']
    best_model = all_results[dataset_name][best_model_name]['Model']
    
    plot_feature_importance(best_model, features, f"{dataset_name} - {best_model_name}", ax)

plt.tight_layout()
plt.show()
## 11. Model Recommendations and Summary# Final recommendations
print("\n" + "="*100)
print("FINAL RECOMMENDATIONS AND SUMMARY")
print("="*100)

print("\n1. BEST PERFORMING MODELS:")
for dataset, best_model in best_models.items():
    print(f"   {dataset:10} -> {best_model['Model']} (R² = {best_model['R²']:.4f})")

print("\n2. DATASET-SPECIFIC INSIGHTS:")
print("   • NaCl Exposure: Shows degradation over time with decreasing compressive strength")
print("   • Water Exposure: Shows strength gain over time, likely due to continued hydration")
print("   • Na2SO4 Exposure: Shows initial strength gain followed by stabilization")

print("\n3. MODEL RECOMMENDATIONS:")
print("   • For high accuracy: Use ensemble methods (Random Forest, XGBoost, LightGBM)")
print("   • For interpretability: Use Linear Regression or Decision Trees")
print("   • For uncertainty quantification: Use Bayesian Ridge or Gaussian Process")
print("   • For novel approaches: Consider Genetic Programming or MARS")

print("\n4. FEATURE IMPORTANCE:")
print("   • Days of exposure is typically the most important feature")
print("   • Fiber properties (length, diameter, percentage) significantly impact strength")
print("   • Water-to-binder ratio affects the concrete matrix properties")

print("\n5. NEXT STEPS:")
print("   • Hyperparameter tuning for best models")
print("   • Feature engineering based on domain knowledge")
print("   • Cross-validation with different strategies")
print("   • Ensemble methods combining multiple models")

# Save results summary
results_summary.to_csv('model_comparison_results.csv', index=False)
print(f"\n6. RESULTS SAVED:")
print("   • Model comparison results saved to 'model_comparison_results.csv'")
print("   • Use this data for further analysis and model selection")
## 12. Hyperparameter Tuning for Best Models# Hyperparameter tuning for the best models
from sklearn.model_selection import GridSearchCV

def tune_best_models(X_train, y_train, best_model_name):
    """Tune hyperparameters for best performing models"""
    
    if best_model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        model = RandomForestRegressor(random_state=42)
    
    elif best_model_name == 'XGBoost':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
        model = xgb.XGBRegressor(random_state=42)
    
    elif best_model_name == 'LightGBM':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [31, 50, 100]
        }
        model = lgb.LGBMRegressor(random_state=42, verbose=-1)
    
    else:
        return None, None
    
    # Perform grid search
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_

# Tune models for each dataset
tuned_models = {}
for dataset in ['NaCl', 'Water', 'Na2SO4', 'Combined']:
    best_model_name = best_models[dataset]['Model']
    
    if dataset == 'NaCl':
        X_train, y_train = X_train_nacl, y_train_nacl
    elif dataset == 'Water':
        X_train, y_train = X_train_water, y_train_water
    elif dataset == 'Na2SO4':
        X_train, y_train = X_train_na2so4, y_train_na2so4
    else:
        X_train, y_train = X_train_combined, y_train_combined
    
    print(f"\nTuning {best_model_name} for {dataset} dataset...")
    tuned_model, best_params = tune_best_models(X_train, y_train, best_model_name)
    
    if tuned_model is not None:
        tuned_models[dataset] = {
            'model': tuned_model,
            'params': best_params,
            'model_name': best_model_name
        }
        print(f"Best parameters: {best_params}")
    else:
        print(f"Hyperparameter tuning not implemented for {best_model_name}")

print("\nHyperparameter tuning completed!")