"""
Simple scikit-learn example using GroupARDRegression with sklearn 1.7 compatibility.
This demonstrates the basic usage with diabetes dataset and RepeatedKFold cross-validation.
"""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.linear_model import BayesianRidge, ARDRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from groupard import GroupARDRegression


def main():
    # Load diabetes dataset
    X, y = load_diabetes(return_X_y=True)
    print(f"Dataset shape: {X.shape}")
    
    # Define two groups of features:
    # Group 0: first 5 features (age, sex, bmi, bp, s1)  
    # Group 1: last 5 features (s2, s3, s4, s5, s6)
    groups = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    
    # Set up cross-validation
    cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
    
    # Define models to test
    models = {
        'ARD Regression': ARDRegression(tol=1e-6),
        'Bayesian Ridge': BayesianRidge(tol=1e-6),
        'GroupARD (2 groups)': GroupARDRegression(
            prior='GroupARD', 
            groups=groups, 
            tol=1e-6
        )
    }
    
    # Run cross-validation
    print("Cross-validation results (R² score):")
    print("-" * 40)
    
    for name, model in models.items():
        # Create pipeline with standardization
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', model)
        ])
        
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='r2')
        mean_score = scores.mean()
        std_score = scores.std()
        
        print(f"{name:20s}: {mean_score:.4f} ± {std_score:.4f}")


if __name__ == "__main__":
    main()

