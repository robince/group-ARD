"""
Complete scikit-learn example using GroupARDRegression with sklearn 1.7 compatibility.

This example demonstrates:
1. Using the diabetes dataset from sklearn
2. Defining logical feature groups
3. Cross-validation with RepeatedKFold
4. Comparing different priors (Ridge, ARD, GroupARD)
"""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.linear_model import BayesianRidge, ARDRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from groupard import GroupARDRegression


def main():
    # Load the diabetes dataset
    print("Loading diabetes dataset...")
    X, y = load_diabetes(return_X_y=True)
    feature_names = load_diabetes().feature_names
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_names}")
    
    # The diabetes dataset has 10 features:
    # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
    # where s1-s6 are six blood serum measurements
    
    # Define feature groups based on logical groupings:
    # Group 0: Demographics (age, sex)
    # Group 1: Physical measurements (bmi, bp) 
    # Group 2: Blood serum measurements (s1, s2, s3, s4, s5, s6)
    
    n_features = X.shape[1]
    print(f"\nCreating feature groups for {n_features} features:")
    
    # Define groups - each feature gets assigned to a group
    feature_groups = np.array([
        0,  # age - demographics
        0,  # sex - demographics  
        1,  # bmi - physical
        1,  # bp - physical
        2,  # s1 - blood serum
        2,  # s2 - blood serum
        2,  # s3 - blood serum
        2,  # s4 - blood serum
        2,  # s5 - blood serum
        2   # s6 - blood serum
    ])
    
    # Print feature group assignments
    for i, (feat, group) in enumerate(zip(feature_names, feature_groups)):
        group_name = ['Demographics', 'Physical', 'Blood_Serum'][group]
        print(f"  {feat}: Group {group} ({group_name})")
    
    # Set up cross-validation
    print(f"\nSetting up cross-validation...")
    cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
    
    # Define models to compare
    models = {
        'Linear Regression': LinearRegression(),
        'Bayesian Ridge': BayesianRidge(tol=1e-6),
        'ARD Regression': ARDRegression(tol=1e-6),
        'GroupARD (Ridge prior)': GroupARDRegression(
            prior='Ridge', 
            tol=1e-6
        ),
        'GroupARD (ARD prior)': GroupARDRegression(
            prior='ARD', 
            tol=1e-6
        ),
        'GroupARD (All same group)': GroupARDRegression(
            prior='GroupARD', 
            groups=np.zeros(n_features, dtype=int),  # All features in one group
            tol=1e-6
        ),
        'GroupARD (Each feature separate)': GroupARDRegression(
            prior='GroupARD', 
            groups=np.arange(n_features),  # Each feature in its own group
            tol=1e-6
        ),
        'GroupARD (Logical groups)': GroupARDRegression(
            prior='GroupARD', 
            groups=feature_groups,  # Our logical groupings
            tol=1e-6
        ),
    }
    
    # Run cross-validation for each model
    print(f"\nRunning cross-validation (5-fold, 10 repeats)...")
    print("=" * 60)
    
    results = {}
    for name, model in models.items():
        print(f"Testing {name}...")
        
        # Create pipeline with standardization
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', model)
        ])
        
        # Run cross-validation
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='r2')
        
        results[name] = scores
        mean_score = scores.mean()
        std_score = scores.std()
        
        print(f"  R² Score: {mean_score:.4f} ± {std_score:.4f}")
    
    print("=" * 60)
    
    # Sort results by mean score
    sorted_results = sorted(results.items(), key=lambda x: x[1].mean(), reverse=True)
    
    print("\nFinal Results (sorted by R² score):")
    print("-" * 50)
    for name, scores in sorted_results:
        mean_score = scores.mean()
        std_score = scores.std()
        print(f"{name:30s}: {mean_score:.4f} ± {std_score:.4f}")
    
    # Demonstrate feature selection with GroupARD
    print(f"\n" + "=" * 60)
    print("Feature Selection Analysis")
    print("=" * 60)
    
    # Fit the best GroupARD model to see which features are selected
    best_model = GroupARDRegression(
        prior='GroupARD', 
        groups=feature_groups,
        tol=1e-6,
        verbose=True
    )
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit the model
    print("\nFitting GroupARD model for feature analysis...")
    best_model.fit(X_scaled, y)
    
    # Analyze coefficients
    print("\nCoefficients and feature importance:")
    
    # Create a list of tuples for analysis
    coef_data = []
    for i, (feat, group, coef) in enumerate(zip(feature_names, feature_groups, best_model.coef_)):
        coef_data.append((feat, group, coef, abs(coef)))
    
    # Sort by absolute coefficient value
    coef_data.sort(key=lambda x: x[3], reverse=True)
    
    # Print table header
    print(f"{'Feature':<12} {'Group':<5} {'Coefficient':<12} {'Abs_Coefficient':<15}")
    print("-" * 50)
    
    # Print data
    for feat, group, coef, abs_coef in coef_data:
        print(f"{feat:<12} {group:<5} {coef:<12.4f} {abs_coef:<15.4f}")
    
    # Show which features were effectively selected (non-zero coefficients)
    selected_features = [(feat, group, coef) for feat, group, coef, abs_coef in coef_data if abs_coef > 1e-6]
    print(f"\nSelected features (|coefficient| > 1e-6): {len(selected_features)}/{len(feature_names)}")
    
    if len(selected_features) > 0:
        print("Selected features:")
        for feat, group, coef in selected_features:
            group_name = ['Demographics', 'Physical', 'Blood_Serum'][group]
            print(f"  {feat:12s} (Group {group} - {group_name}): {coef:8.4f}")
    
    # Group-level analysis
    print("\nGroup-level coefficient analysis:")
    group_names = ['Demographics', 'Physical', 'Blood_Serum']
    for group_id, group_name in enumerate(group_names):
        group_mask = feature_groups == group_id
        group_coefs = best_model.coef_[group_mask]
        group_sum = np.sum(np.abs(group_coefs))
        active_features = np.sum(np.abs(group_coefs) > 1e-6)
        total_features = np.sum(group_mask)
        
        print(f"  {group_name:15s}: {active_features}/{total_features} features active, "
              f"total |coef| = {group_sum:.4f}")


if __name__ == "__main__":
    main()