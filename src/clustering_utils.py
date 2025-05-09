import pandas as pd
import numpy as np
from functools import reduce

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import Normalizer

from config import *

#########################
### Feature selection ###
#########################

# Remove low variance features
def remove_low_variance_features(X, threshold=0.01):
    """
    Remove features with variance below a specified threshold.
    
    Parameters:
    -----------
    X : DataFrame
        Feature matrix
    threshold : float, optional (default=0.01)
        Variance threshold below which features will be removed
        
    Returns:
    --------
    X_reduced : DataFrame
        Feature matrix with low variance features removed
    """
    # Normalize data
    for column in X.columns:
        X[column] = (X[column] - X[column].min()) / (X[column].max() - X[column].min())
    
    # Apply variance threshold
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)

    # Get selected feature names
    selected_feature_names = X.columns[selector.get_support()]

    # Create reduced DataFrame
    X_reduced = X[selected_feature_names]

    logging.info(f"Kept {X_reduced.shape[1]} features out of {X.shape[1]}")
    
    return X_reduced

# Remove highly correlated features
def remove_multicollineraity(X, y, feature_corr_threshold=0.85,verbose=True):
    """
    Remove highly correlated features while prioritizing features with higher correlation to target.
    
    Parameters:
    -----------
    X : DataFrame
        Feature matrix
    y : Series or DataFrame
        Target variable
    feature_corr_threshold : float, optional (default=0.85)
        Threshold above which features are considered highly correlated
    verbose : bool, optional (default=True)
        Whether to print information about dropped features
        
    Returns:
    --------
    X_reduced : DataFrame
        Feature matrix with highly correlated features removed
    """
    # Covert y to series
    y = y.iloc[:,0]
    
    # Calculate feature-feature correlation matrix
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Calculate feature-target correlation
    corr_with_y = pd.DataFrame(X.corrwith(y).abs(), columns=["Corr"])
    
    # Add variance as secondary criterion for tie-breaking
    corr_with_y["Variance"] = X.var().values
    
    # Sort by correlation with target (primary) and variance (secondary)
    corr_with_y_sorted = corr_with_y.sort_values(
        by=["Corr", "Variance"], ascending=[False, False]
    )

    # Remove highly correlated features
    to_drop = set()
    for column in corr_with_y_sorted.index:
        if column in to_drop:
            continue
        
        # Find highly correlated features
        corr_features = upper.index[upper[column] > 0.85].tolist()
        
        for corr_feature in corr_features:
            if corr_feature not in to_drop:
                to_drop.add(corr_feature)

    X_reduced = X.drop(columns=list(to_drop))

    logging.info(f"Removed {len(to_drop)} highly correlated features: {to_drop}")
    
    return X_reduced