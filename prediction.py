import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.compose import TransformedTargetRegressor
from scipy import stats

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
OUTLIER_THRESHOLD = 3

def load_data(path="housing.csv"):
    """Load and preprocess housing data"""
    df = pd.read_csv(path)
    
    # Handle capped values and apply log transform
    df["median_house_value"] = np.log1p(df["median_house_value"])
    
    # Feature engineering
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]
    
    # Handle missing values (Fixed Pandas 3.0 Warning)
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ["float64", "int64"]:
                df[col] = df[col].fillna(df[col].median())  # ✅ Safe replacement
            else:
                df[col] = df[col].fillna(df[col].mode()[0])  # ✅ Safe replacement
    
    # Convert categorical variables into dummy variables
    df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)
    
    return df

def perform_eda(df):
    """Generate exploratory data analysis visualizations"""
    plt.figure(figsize=(18, 12))
    
    # Distribution of target variable
    plt.subplot(2, 2, 1)
    sns.histplot(df["median_house_value"], kde=True, bins=30)
    plt.title("House Value Distribution")
    plt.xlabel("Median House Value (log scale)")
    
    # Correlation heatmap
    plt.subplot(2, 2, 2)
    corr_matrix = df.corr(numeric_only=True)
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", cbar_kws={"label": "Correlation Coefficient"})
    plt.title("Feature Correlation Matrix")
    
    # Geographic distribution
    plt.subplot(2, 2, 3)
    sns.scatterplot(x="longitude", y="latitude", data=df, hue="median_house_value", palette="viridis",
                    size="population", sizes=(10, 200), alpha=0.5)
    plt.title("Geographic Price Distribution")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    
    # Feature relationships
    plt.subplot(2, 2, 4)
    sns.pairplot(df[["median_house_value", "median_income", "housing_median_age", "rooms_per_household"]])
    plt.tight_layout()
    plt.show()

def preprocess_data(df):
    """Handle outliers and prepare data for modeling"""
    numeric_cols = df.select_dtypes(include=np.number).columns
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    df_clean = df[(z_scores < OUTLIER_THRESHOLD).all(axis=1)]
    
    X = df_clean.drop("median_house_value", axis=1)
    y = df_clean["median_house_value"]
    
    return X, y

def train_models(X, y):
    """Train and compare multiple regression models"""
    models = {
        "Linear Regression": make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=2, include_bias=False),
            LinearRegression()
        ),
        "Lasso Regression": make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=2, include_bias=False),
            Lasso(random_state=RANDOM_STATE, max_iter=10000)
        ),
        "Random Forest": RandomForestRegressor(
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=RANDOM_STATE
        )
    }
    
    param_grids = {
        "Lasso Regression": {"lasso__alpha": [0.01, 0.1, 1, 10]},
        "Random Forest": {"n_estimators": [100, 200], "max_depth": [None, 10]},
        "Gradient Boosting": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1]}
    }
    
    best_models = {}
    for name, model in models.items():
        if name in param_grids:
            gs = GridSearchCV(
                model, 
                param_grids[name], 
                cv=5,
                scoring="neg_mean_squared_error", 
                n_jobs=-1
            )
            gs.fit(X, y)
            best_models[name] = gs.best_estimator_
        else:
            model.fit(X, y)
            best_models[name] = model
    
    return best_models

def evaluate_models(models, X_test, y_test):
    """Generate comprehensive model evaluation report"""
    metrics = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        metrics.append({
            "Model": name,
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "MAE": mean_absolute_error(y_test, y_pred),
            "R²": r2_score(y_test, y_pred),
            "Cross-Val R²": cross_val_score(model, X_test, y_test, cv=5).mean()
        })
    
    metrics_df = pd.DataFrame(metrics)
    print("🏆 Model Performance Comparison:")
    print(metrics_df.sort_values("RMSE"))
    
    # Feature importance visualization
    plt.figure(figsize=(12, 8))
    feature_importance = permutation_importance(
        models["Gradient Boosting"], 
        X_test, 
        y_test, 
        n_repeats=10,
        random_state=RANDOM_STATE, 
        n_jobs=-1
    )
    sorted_idx = feature_importance.importances_mean.argsort()[::-1]
    plt.barh(
        X_test.columns[sorted_idx], 
        feature_importance.importances_mean[sorted_idx]
    )
    plt.xlabel("Permutation Importance")
    plt.title("Feature Importance Analysis")
    plt.tight_layout()
    plt.show()
    
    return metrics_df

if __name__ == "__main__":
    # Data pipeline
    df = load_data()
    perform_eda(df)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )
    
    # Model training
    models = train_models(X_train, y_train)
    
    # Evaluation
    metrics_df = evaluate_models(models, X_test, y_test)
    
    # Save best model
    best_model = models[metrics_df.loc[metrics_df["RMSE"].idxmin(), "Model"]]
    joblib.dump(best_model, "best_housing_model.pkl")
    
    # Final diagnostics
    final_model = TransformedTargetRegressor(
        regressor=best_model,
        func=np.log1p,
        inverse_func=np.expm1
    )
    final_model.fit(X_train, y_train)
    
    print(f"\n💾 Best model saved as 'best_housing_model.pkl'")
    print("\n🔍 Final Model Diagnostics:")
    print(f"Training Score: {final_model.score(X_train, y_train):.2f}")
    print(f"Testing Score: {final_model.score(X_test, y_test):.2f}")

