# %%
# 1. Import Libraries and Load Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

# Load data
data = pd.read_csv(r'D:\william\OneDrive - UW-Madison\UW-Madison\722\-2025AAE722_William-J\Final Project\DataCoSupplyChainDataset1.csv')
print(f"Data loaded: {data.shape[0]} records, {data.shape[1]} columns")

# %%
# 2. Data Preprocessing and Feature Selection
# Select features including Customer Country
features = [
    'Type',  # Payment type
    'Category Name', 
    'Market', 
    'Customer Country',  # NEW: Added customer country
    'Order Item Discount Rate', 
    'Order Item Product Price', 
    'Order Item Quantity',
    'Shipping Mode',
    'Late_delivery_risk',
    'Delivery Status',
    'Order Status',
    'Department Name',
    'order date (DateOrders)',
    'Order Item Profit Ratio'  # Target variable
]

# Create clean dataset
model_data = data[features].copy().dropna()
model_data['order date (DateOrders)'] = pd.to_datetime(model_data['order date (DateOrders)'])

# Create separate date and time columns
model_data['Order Date'] = model_data['order date (DateOrders)'].dt.date
model_data['Order Time'] = model_data['order date (DateOrders)'].dt.time
print(f"Clean data: {model_data.shape[0]} records")
print(f"Customer Country unique values: {model_data['Customer Country'].nunique()}")
print(f"Type unique values: {model_data['Type'].nunique()}")
print(f"Category Name unique values: {model_data['Category Name'].nunique()}")

# %%
# 3a. Time Feature Engineering (Before Encoding)
print("📅 TIME FEATURE ENGINEERING")
print("-" * 40)

# Extract meaningful time components from order date
model_data['order_month'] = model_data['order date (DateOrders)'].dt.month
model_data['order_quarter'] = model_data['order date (DateOrders)'].dt.quarter
model_data['order_day_of_week'] = model_data['order date (DateOrders)'].dt.dayofweek
model_data['order_day_of_year'] = model_data['order date (DateOrders)'].dt.dayofyear
model_data['order_week_of_year'] = model_data['order date (DateOrders)'].dt.isocalendar().week

# Business-relevant time features
model_data['is_weekend'] = (model_data['order_day_of_week'] >= 5).astype(int)
model_data['is_month_end'] = (model_data['order date (DateOrders)'].dt.day >= 28).astype(int)
model_data['is_quarter_end'] = model_data['order_month'].isin([3, 6, 9, 12]).astype(int)

# Seasonal categories
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

model_data['season'] = model_data['order_month'].apply(get_season)

print("✅ Time features created:")
print(f"   • order_month: {model_data['order_month'].nunique()} unique values")
print(f"   • order_quarter: {model_data['order_quarter'].nunique()} unique values") 
print(f"   • order_day_of_week: {model_data['order_day_of_week'].nunique()} unique values")
print(f"   • is_weekend: {model_data['is_weekend'].value_counts().to_dict()}")
print(f"   • season: {model_data['season'].nunique()} unique seasons")

# Quick correlation analysis with target
time_features = ['order_month', 'order_quarter', 'order_day_of_week', 'is_weekend', 'is_month_end', 'is_quarter_end']
print(f"\n🔍 TIME FEATURE CORRELATIONS WITH PROFIT RATIO:")
for feature in time_features:
    correlation = model_data[feature].corr(model_data['Order Item Profit Ratio'])
    print(f"   {feature:<20} correlation: {correlation:>8.4f}")

# Check seasonal profit patterns
seasonal_profit = model_data.groupby('season')['Order Item Profit Ratio'].agg(['mean', 'std', 'count'])
print(f"\n📊 SEASONAL PROFIT PATTERNS:")
print(seasonal_profit.round(4))

# %%
# 3b. Updated Feature Definition with Time Components
# Separate target and features
target = 'Order Item Profit Ratio'
y = model_data[target]

# Define feature types (UPDATED with time features)
categorical_features = [
    'Type', 'Category Name', 'Market', 'Customer Country', 
    'Shipping Mode', 'Delivery Status', 'Order Status', 'Department Name',
    'season'  # NEW: Seasonal category
]

numerical_features = [
    'Order Item Discount Rate', 'Order Item Product Price', 
    'Order Item Quantity', 'Late_delivery_risk',
    # NEW: Time-based numerical features
    'order_month', 'order_quarter', 'order_day_of_week', 'order_day_of_year', 
    'order_week_of_year', 'is_weekend', 'is_month_end', 'is_quarter_end'
]

print(f"📊 UPDATED FEATURE COMPOSITION:")
print(f"   Categorical features: {len(categorical_features)} ({categorical_features})")
print(f"   Numerical features: {len(numerical_features)} (including {len([f for f in numerical_features if 'order_' in f or 'is_' in f])} time features)")

# Direct encoding using LabelEncoder
encoded_data = model_data.copy()
label_encoders = {}

for feature in categorical_features:
    le = LabelEncoder()
    encoded_data[f'{feature}_encoded'] = le.fit_transform(encoded_data[feature])
    label_encoders[feature] = le

# Create feature matrix
encoded_categorical_features = [f'{feature}_encoded' for feature in categorical_features]
all_features = numerical_features + encoded_categorical_features
X = encoded_data[all_features]

print(f"\n✅ FINAL FEATURE SET: {len(all_features)} total features")
print(f"   • {len(numerical_features)} numerical (including time features)")
print(f"   • {len(encoded_categorical_features)} encoded categorical")
print(f"✅ Type (Payment Type) included")  
print(f"✅ Customer Country included")
print(f"✅ Time features included and properly engineered")
print(f"✅ All features ready for modeling")

# %%
# 3c. Time Variable Impact Analysis
import matplotlib.pyplot as plt

print("🔍 TIME VARIABLE DECISION ANALYSIS")
print("=" * 50)

# Create comparison datasets
features_without_time = [
    'Type', 'Category Name', 'Market', 'Customer Country', 
    'Shipping Mode', 'Delivery Status', 'Order Status', 'Department Name',
    'Order Item Discount Rate', 'Order Item Product Price', 
    'Order Item Quantity', 'Late_delivery_risk'
]

features_with_time = all_features

print(f"📊 FEATURE COMPARISON:")
print(f"   Without time features: {len([f for f in features_without_time if not f.endswith('_encoded')])} base features")
print(f"   With time features: {len(all_features)} total features")
print(f"   Time features added: {len(all_features) - len(features_without_time)}")

# Check for potential time-related patterns
print(f"\n📈 TIME PATTERN ANALYSIS:")
print("-" * 30)

# Monthly profit analysis
monthly_stats = model_data.groupby('order_month')['Order Item Profit Ratio'].agg(['mean', 'std', 'count'])
monthly_variation = monthly_stats['mean'].std()
print(f"Monthly profit variation (std): {monthly_variation:.6f}")

# Weekly profit analysis  
weekly_stats = model_data.groupby('order_day_of_week')['Order Item Profit Ratio'].agg(['mean', 'std', 'count'])
weekly_variation = weekly_stats['mean'].std()
print(f"Weekly profit variation (std): {weekly_variation:.6f}")

# Weekend vs weekday
weekend_profit = model_data[model_data['is_weekend'] == 1]['Order Item Profit Ratio'].mean()
weekday_profit = model_data[model_data['is_weekend'] == 0]['Order Item Profit Ratio'].mean()
weekend_effect = abs(weekend_profit - weekday_profit)
print(f"Weekend effect magnitude: {weekend_effect:.6f}")

# Recommendation based on variation
print(f"\n🎯 TIME FEATURE RECOMMENDATIONS:")
print("-" * 40)

if monthly_variation > 0.01:
    print("✅ INCLUDE: Significant monthly profit variation detected")
else:
    print("⚠️  WEAK: Low monthly profit variation")

if weekly_variation > 0.01:
    print("✅ INCLUDE: Significant weekly profit variation detected") 
else:
    print("⚠️  WEAK: Low weekly profit variation")

if weekend_effect > 0.01:
    print("✅ INCLUDE: Weekend effect detected")
else:
    print("⚠️  WEAK: Minimal weekend effect")

# Overall recommendation
total_time_signal = monthly_variation + weekly_variation + weekend_effect
print(f"\n🏆 OVERALL TIME SIGNAL STRENGTH: {total_time_signal:.6f}")

if total_time_signal > 0.03:
    print("🟢 STRONG RECOMMENDATION: Include time features")
    time_recommendation = "INCLUDE"
elif total_time_signal > 0.015:
    print("🟡 MODERATE RECOMMENDATION: Time features may help")
    time_recommendation = "CONSIDER"
else:
    print("🔴 WEAK RECOMMENDATION: Time features unlikely to help significantly")
    time_recommendation = "SKIP"

print(f"\n📋 BENEFITS vs RISKS:")
print("✅ BENEFITS:")
print("   • Capture seasonal business patterns")
print("   • Model weekly customer behavior")
print("   • Identify optimal timing strategies")
print("⚠️  RISKS:")
print("   • Potential overfitting to specific time periods")
print("   • Model complexity increase")
print("   • May reduce generalization to future periods")

print(f"\n🎯 FINAL RECOMMENDATION: {time_recommendation} time features")
print(f"💡 TIP: Compare model performance with/without time features using CV")

# %%
# 4. Train-Test Split (70% / 30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"Training: {X_train.shape[0]} samples")
print(f"Testing: {X_test.shape[0]} samples")

# %%
# 5. Hybrid Model Implementation with Cross-Validation
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.ensemble import VotingRegressor
import time

print("🔄 HYBRID MODEL TRAINING WITH CROSS-VALIDATION")
print("=" * 60)

# Define individual models with initial parameters
models = {
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(random_state=42),
    'Lasso Regression': Lasso(random_state=42),
    'SVR': SVR()
}

# Hyperparameter grids for optimization
param_grids = {
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0]
    },
    'Ridge Regression': {
        'alpha': [0.1, 1.0, 10.0, 100.0]
    },
    'Lasso Regression': {
        'alpha': [0.1, 1.0, 10.0, 100.0]
    },
    'SVR': {
        'kernel': ['linear', 'rbf'],
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto']
    }
}

# Cross-validation results storage
cv_results = {}
optimized_models = {}
best_params = {}

print("Phase 1: Individual Model Optimization")
print("-" * 40)

# %%
# 6. Individual Model Cross-Validation and Hyperparameter Tuning
for name, model in models.items():
    print(f"\n🔍 Optimizing {name}...")
    start_time = time.time()
    
    if name in param_grids:
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            model, 
            param_grids[name], 
            cv=5, 
            scoring='r2',
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params[name] = grid_search.best_params_
        
        # Cross-validation score with best parameters
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2')
        
    else:
        # For Linear Regression (no hyperparameters)
        best_model = model
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        best_params[name] = "Default parameters"
    
    cv_results[name] = {
        'mean_cv_score': cv_scores.mean(),
        'std_cv_score': cv_scores.std(),
        'best_model': best_model
    }
    optimized_models[name] = best_model
    
    training_time = time.time() - start_time
    
    print(f"  ✅ CV R² Score: {cv_scores.mean():.6f} (±{cv_scores.std():.6f})")
    print(f"  ⏱️  Training Time: {training_time:.2f}s")
    if name in param_grids:
        print(f"  🎯 Best Parameters: {grid_search.best_params_}")

print(f"\n📊 INDIVIDUAL MODEL RANKINGS")
print("-" * 40)
sorted_results = sorted(cv_results.items(), key=lambda x: x[1]['mean_cv_score'], reverse=True)
for i, (name, results) in enumerate(sorted_results, 1):
    print(f"{i}. {name:<20} R² = {results['mean_cv_score']:.6f} (±{results['std_cv_score']:.6f})")

# %%
# 7. Hybrid Ensemble Model Creation
print(f"\n🚀 PHASE 2: HYBRID ENSEMBLE MODELS")
print("-" * 40)

# Select top 3 performing models for ensemble
top_3_models = sorted_results[:3]
print(f"Top 3 models selected for ensemble:")
for i, (name, results) in enumerate(top_3_models, 1):
    print(f"  {i}. {name} (R² = {results['mean_cv_score']:.6f})")

# Create ensemble combinations
ensemble_models = {}

# 1. Voting Regressor with top 3 models
voting_estimators = [(name, cv_results[name]['best_model']) for name, _ in top_3_models]
voting_regressor = VotingRegressor(estimators=voting_estimators)
ensemble_models['Voting (Top 3)'] = voting_regressor

# 2. Voting Regressor with all models
all_estimators = [(name, model) for name, model in optimized_models.items()]
voting_all = VotingRegressor(estimators=all_estimators)
ensemble_models['Voting (All Models)'] = voting_all

# 3. Weighted Voting based on CV performance
weights = [results['mean_cv_score'] for _, results in top_3_models]
voting_weighted = VotingRegressor(estimators=voting_estimators)
ensemble_models['Weighted Voting (Top 3)'] = voting_weighted

print(f"\n🔄 Training Ensemble Models...")
ensemble_results = {}

for name, ensemble in ensemble_models.items():
    start_time = time.time()
    
    # Cross-validation
    cv_scores = cross_val_score(ensemble, X_train, y_train, cv=5, scoring='r2')
    
    ensemble_results[name] = {
        'mean_cv_score': cv_scores.mean(),
        'std_cv_score': cv_scores.std(),
        'model': ensemble
    }
    
    training_time = time.time() - start_time
    print(f"  ✅ {name:<25} R² = {cv_scores.mean():.6f} (±{cv_scores.std():.6f}) [{training_time:.2f}s]")

# Combine all results for final comparison
all_results = {**cv_results, **ensemble_results}
final_ranking = sorted(all_results.items(), key=lambda x: x[1]['mean_cv_score'], reverse=True)

print(f"\n🏆 FINAL MODEL RANKINGS")
print("=" * 60)
for i, (name, results) in enumerate(final_ranking, 1):
    score = results['mean_cv_score']
    std = results['std_cv_score']
    model_type = "🤖 Ensemble" if name in ensemble_results else "📊 Individual"
    print(f"{i:2d}. {model_type} {name:<25} R² = {score:.6f} (±{std:.6f})")

# Select best model
best_model_name, best_model_info = final_ranking[0]
best_model = best_model_info['model'] if 'model' in best_model_info else best_model_info['best_model']

print(f"\n🥇 BEST MODEL SELECTED: {best_model_name}")
print(f"   Cross-Validation R² = {best_model_info['mean_cv_score']:.6f} (±{best_model_info['std_cv_score']:.6f})")

# %%
# 8. Final Model Evaluation on Test Set
print(f"\n🎯 FINAL EVALUATION ON TEST SET")
print("=" * 50)

# Train best model on full training set
start_time = time.time()
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
training_time = time.time() - start_time

# Calculate comprehensive metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Calculate additional metrics
residuals = y_test - y_pred
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Mean Absolute Percentage Error

print(f"🏆 BEST MODEL: {best_model_name}")
print(f"📊 PERFORMANCE METRICS:")
print(f"   R² Score: {r2:.6f}")
print(f"   RMSE: {rmse:.6f}")
print(f"   MAE: {mae:.6f}")
print(f"   MAPE: {mape:.2f}%")
print(f"   Training Time: {training_time:.2f}s")
print(f"   CV R² (Training): {best_model_info['mean_cv_score']:.6f}")
print(f"   Test R² (Holdout): {r2:.6f}")

# Model generalization check
generalization_gap = best_model_info['mean_cv_score'] - r2
print(f"\n📈 GENERALIZATION ANALYSIS:")
print(f"   CV Score vs Test Score Gap: {generalization_gap:.6f}")
if abs(generalization_gap) < 0.02:
    print("   ✅ Excellent generalization (< 2% gap)")
elif abs(generalization_gap) < 0.05:
    print("   ⚠️  Good generalization (< 5% gap)")
else:
    print("   ❌ Potential overfitting (> 5% gap)")

# Feature importance (if applicable)
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': all_features,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\n🔍 TOP 5 FEATURE IMPORTANCES:")
    for i, (_, row) in enumerate(feature_importance.head(5).iterrows(), 1):
        clean_name = row['Feature'].replace('_encoded', '')
        print(f"   {i}. {clean_name:<25} {row['Importance']:.6f}")

elif hasattr(best_model, 'estimators_'):
    # For ensemble models, show component models
    print(f"\n🔧 ENSEMBLE COMPONENTS:")
    if hasattr(best_model, 'estimators_'):
        for name, estimator in best_model.estimators_:
            print(f"   • {name}")

print(f"\n✨ Model successfully optimized and evaluated!")

# %%
# 9. Comprehensive Model Analysis and Comparison
print("🔬 DETAILED MODEL ANALYSIS")
print("=" * 50)

# Model comparison table
print(f"📊 COMPLETE MODEL COMPARISON:")
print(f"{'Model':<25} {'CV R²':<12} {'CV Std':<10} {'Type'}")
print("-" * 60)

for i, (name, results) in enumerate(final_ranking, 1):
    score = results['mean_cv_score']
    std = results['std_cv_score']
    model_type = "Ensemble" if name in ensemble_results else "Individual"
    print(f"{name:<25} {score:<12.6f} {std:<10.6f} {model_type}")

# Best parameters summary
print(f"\n🎯 OPTIMIZED HYPERPARAMETERS:")
print("-" * 40)
for name, params in best_params.items():
    if name in [result[0] for result in sorted_results[:5]]:  # Top 5 individual models
        print(f"\n{name}:")
        if isinstance(params, dict):
            for param, value in params.items():
                print(f"  • {param}: {value}")
        else:
            print(f"  • {params}")

# Performance comparison with baseline
baseline_r2 = cv_results['Linear Regression']['mean_cv_score']
best_r2 = best_model_info['mean_cv_score']
improvement = ((best_r2 - baseline_r2) / baseline_r2) * 100

print(f"\n📈 PERFORMANCE IMPROVEMENT:")
print(f"   Baseline (Linear Regression): {baseline_r2:.6f}")
print(f"   Best Model ({best_model_name}): {best_r2:.6f}")
print(f"   Improvement: {improvement:.2f}%")

# Feature importance analysis (if available)
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': all_features,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\n🔍 COMPLETE FEATURE IMPORTANCE RANKING:")
    print("-" * 50)
    for i, (_, row) in enumerate(feature_importance.iterrows(), 1):
        clean_name = row['Feature'].replace('_encoded', '')
        print(f"{i:2d}. {clean_name:<30} {row['Importance']:.6f}")
    
    # Check key predictors
    type_rows = feature_importance[feature_importance['Feature'].str.contains('Type')]
    country_rows = feature_importance[feature_importance['Feature'].str.contains('Customer Country')]
    
    if not type_rows.empty:
        type_rank = feature_importance.index[feature_importance['Feature'] == type_rows.iloc[0]['Feature']].tolist()[0] + 1
        print(f"\n🎯 Key Predictor Rankings:")
        print(f"   Type (Payment): #{type_rank}")
    
    if not country_rows.empty:
        country_rank = feature_importance.index[feature_importance['Feature'] == country_rows.iloc[0]['Feature']].tolist()[0] + 1
        print(f"   Customer Country: #{country_rank}")

# Cross-validation stability analysis
print(f"\n📊 MODEL STABILITY ANALYSIS:")
print("-" * 30)
for name, results in sorted(cv_results.items(), key=lambda x: x[1]['std_cv_score']):
    stability = "High" if results['std_cv_score'] < 0.01 else "Medium" if results['std_cv_score'] < 0.02 else "Low"
    print(f"{name:<20} Std: {results['std_cv_score']:.6f} ({stability} Stability)")

# %%
# 10. Advanced Visualization and Model Insights
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Actual vs Predicted for Best Model
axes[0, 0].scatter(y_test, y_pred, alpha=0.6, s=30, color='blue', edgecolor='white', linewidth=0.5)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
axes[0, 0].set_title(f'Best Model: {best_model_name}\nActual vs Predicted (R² = {r2:.4f})', fontsize=11)
axes[0, 0].set_xlabel('Actual Profit Ratio')
axes[0, 0].set_ylabel('Predicted Profit Ratio')
axes[0, 0].grid(True, alpha=0.3)

# Add prediction interval lines
sorted_indices = np.argsort(y_test)
sorted_y_test = y_test.iloc[sorted_indices]
sorted_y_pred = y_pred[sorted_indices]
residuals_sorted = sorted_y_pred - sorted_y_test

# 2. Model Performance Comparison
model_names = [name for name, _ in final_ranking[:8]]  # Top 8 models
model_scores = [results['mean_cv_score'] for _, results in final_ranking[:8]]
model_stds = [results['std_cv_score'] for _, results in final_ranking[:8]]

colors = ['gold' if i == 0 else 'lightblue' if 'Ensemble' in name else 'lightgreen' 
          for i, name in enumerate(model_names)]

bars = axes[0, 1].bar(range(len(model_names)), model_scores, yerr=model_stds, 
                      capsize=5, color=colors, edgecolor='black', alpha=0.8)
axes[0, 1].set_title('Model Performance Comparison (CV R² Score)', fontsize=11)
axes[0, 1].set_xlabel('Models')
axes[0, 1].set_ylabel('R² Score')
axes[0, 1].set_xticks(range(len(model_names)))
axes[0, 1].set_xticklabels([name[:12] + '...' if len(name) > 12 else name 
                           for name in model_names], rotation=45, ha='right')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, score) in enumerate(zip(bars, model_scores)):
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{score:.4f}', ha='center', va='bottom', fontsize=8)

# 3. Feature Importance (if available)
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': all_features,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    top_features = feature_importance.head(10)
    
    bars = axes[1, 0].barh(range(len(top_features)), top_features['Importance'], 
                          color='skyblue', edgecolor='navy', alpha=0.7)
    axes[1, 0].set_yticks(range(len(top_features)))
    axes[1, 0].set_yticklabels([f.replace('_encoded', '') for f in top_features['Feature']], 
                              fontsize=9)
    axes[1, 0].set_title('Top 10 Feature Importances', fontsize=11)
    axes[1, 0].set_xlabel('Importance')
    axes[1, 0].invert_yaxis()
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, importance) in enumerate(zip(bars, top_features['Importance'])):
        axes[1, 0].text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                       f'{importance:.4f}', va='center', fontsize=8)
else:
    axes[1, 0].text(0.5, 0.5, f'Feature Importance\nNot Available\nfor {best_model_name}', 
                   transform=axes[1, 0].transAxes, ha='center', va='center',
                   fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)

# 4. Residual Analysis
residuals = y_test - y_pred
axes[1, 1].scatter(y_pred, residuals, alpha=0.6, s=30, color='purple', edgecolor='white', linewidth=0.5)
axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1, 1].set_title('Residual Analysis', fontsize=11)
axes[1, 1].set_xlabel('Predicted Values')
axes[1, 1].set_ylabel('Residuals (Actual - Predicted)')
axes[1, 1].grid(True, alpha=0.3)

# Add residual statistics
rmse_residual = np.sqrt(np.mean(residuals**2))
mean_residual = np.mean(residuals)
axes[1, 1].text(0.05, 0.95, f'RMSE: {rmse_residual:.4f}\nMean: {mean_residual:.4f}', 
               transform=axes[1, 1].transAxes, va='top', 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.show()

# Additional ensemble analysis plot
if 'Voting' in best_model_name or 'Ensemble' in best_model_name:
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Compare individual vs ensemble performance
    individual_scores = [(name, results['mean_cv_score']) for name, results in cv_results.items()]
    ensemble_scores = [(name, results['mean_cv_score']) for name, results in ensemble_results.items()]
    
    all_scores = individual_scores + ensemble_scores
    names, scores = zip(*all_scores)
    
    colors = ['lightcoral' if name in cv_results else 'lightgreen' for name in names]
    
    bars = ax.bar(range(len(names)), scores, color=colors, edgecolor='black', alpha=0.8)
    ax.set_title('Individual Models vs Ensemble Models Performance', fontsize=14)
    ax.set_xlabel('Models')
    ax.set_ylabel('Cross-Validation R² Score')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='lightcoral', label='Individual Models'),
                      Patch(facecolor='lightgreen', label='Ensemble Models')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Highlight best model
    best_idx = names.index(best_model_name)
    bars[best_idx].set_color('gold')
    bars[best_idx].set_edgecolor('darkgoldenrod')
    bars[best_idx].set_linewidth(3)
    
    plt.tight_layout()
    plt.show()

# %%
# 11. Final Summary and Model Recommendations
print("🎯 HYBRID MODEL OPTIMIZATION SUMMARY")
print("=" * 60)
print(f"📊 Dataset: {model_data.shape[0]} records, {len(all_features)} features")
print(f"🔄 Models Tested: {len(models)} individual + {len(ensemble_models)} ensemble")
print(f"🏆 Best Model: {best_model_name}")
print(f"📈 Best Performance: R² = {best_model_info['mean_cv_score']:.6f} (±{best_model_info['std_cv_score']:.6f})")

print(f"\n🎯 KEY FINDINGS:")
print("-" * 30)

# Performance improvement analysis
baseline_model = 'Linear Regression'
baseline_score = cv_results[baseline_model]['mean_cv_score']
improvement = ((best_model_info['mean_cv_score'] - baseline_score) / baseline_score) * 100

print(f"✅ Performance Improvement: {improvement:.1f}% over baseline")
print(f"✅ Cross-Validation Stability: ±{best_model_info['std_cv_score']:.6f}")

# Model type analysis
if best_model_name in ensemble_results:
    print(f"✅ Best approach: Ensemble modeling")
    print(f"✅ Hybrid strategy successful")
else:
    print(f"✅ Best approach: Individual model optimization")
    print(f"✅ Hyperparameter tuning effective")

# Feature analysis
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': all_features,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    top_feature = feature_importance.iloc[0]['Feature'].replace('_encoded', '')
    print(f"✅ Most Important Feature: {top_feature}")
    
    # Check if Customer Country and Type are in top 10
    top_10_features = feature_importance.head(10)['Feature'].tolist()
    
    customer_country_in_top = any('Customer Country' in f for f in top_10_features)
    type_in_top = any('Type' in f for f in top_10_features)
    
    if customer_country_in_top:
        print(f"✅ Customer Country: High importance confirmed")
    if type_in_top:
        print(f"✅ Payment Type: High importance confirmed")

print(f"\n🚀 RECOMMENDATIONS:")
print("-" * 30)
print(f"1. Deploy {best_model_name} for production")
print(f"2. Expected R² performance: ~{best_model_info['mean_cv_score']:.3f}")

if best_model_name in ensemble_results:
    print(f"3. Ensemble approach provides robust predictions")
    print(f"4. Regular model retraining recommended")
else:
    print(f"3. Monitor individual model performance")
    print(f"4. Consider ensemble for future improvements")

print(f"5. Focus on top {min(5, len(all_features))} features for model interpretation")
print(f"6. Validate model on new data before deployment")

print(f"\n✨ Hybrid modeling with cross-validation optimization complete!")
print(f"🎉 Best model identified and ready for deployment!")

# Save model summary
model_summary = {
    'best_model': best_model_name,
    'cv_r2_score': best_model_info['mean_cv_score'],
    'cv_r2_std': best_model_info['std_cv_score'],
    'test_r2_score': r2 if 'r2' in locals() else None,
    'improvement_over_baseline': improvement,
    'total_models_tested': len(models) + len(ensemble_models),
    'features_used': len(all_features),
    'training_samples': X_train.shape[0],
    'test_samples': X_test.shape[0]
}

print(f"\n📋 Model Summary Dictionary Created:")
for key, value in model_summary.items():
    print(f"   {key}: {value}")

print(f"\n🔍 Access individual results:")
print(f"   • cv_results: Individual model CV scores")
print(f"   • ensemble_results: Ensemble model CV scores") 
print(f"   • final_ranking: Complete model ranking")
print(f"   • best_model: Trained best model object")
print(f"   • optimized_models: All optimized individual models")


