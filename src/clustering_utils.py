import pandas as pd
import numpy as np
from functools import reduce

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import Normalizer

# import umap

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
        corr_features = upper.index[upper[column] > feature_corr_threshold].tolist()
        
        for corr_feature in corr_features:
            if corr_feature not in to_drop:
                to_drop.add(corr_feature)

    X_reduced = X.drop(columns=list(to_drop))

    logging.info(f"Removed {len(to_drop)} highly correlated features: {to_drop}")
    
    return X_reduced

def remove_outliers(X, method:str = "MAD", threshold:float = 3.0):
    if method == 'MAD':
        for col in X.columns:
            data = X[col]
            median = np.nanmedian(data)
            abs_diff = np.abs(data - median)
            mad = np.nanmedian(abs_diff)

            if mad == 0 or np.isnan(mad):
                continue

            outlier_mask = (np.abs(data - median)) > (threshold * mad)
            X.loc[outlier_mask, col] = np.nan
        return X
    else:
        return X


################
###   Plot   ###
################

 
def plot_scatter_dimensionality_reduction(X, reducer=None, labels=None , arrows:bool=True, ax=None):
    coeff = None
    n_features_provided = X.shape[1]
    if reducer is not None and n_features_provided > 2:
        X_reduced = reducer.fit_transform(X)
        if "PCA" in reducer.__str__():
            coeff = reducer.components_
        X_reduced_df = pd.DataFrame({"Component 1": X_reduced[:,0]
                                ,'Component 2': X_reduced[:,1]})
    else:
        if n_features_provided != 2:
            raise ValueError('Dimensionality reduction method must be provided when number of features are not 2')
        X_reduced_df = X

    x_var_name, y_var_name = X_reduced_df.columns

    sns.scatterplot(X_reduced_df, x=x_var_name, y=y_var_name,ax=ax)
    if "umap" in reducer.__str__().lower():
        title_reducer = "UMAP()"
    else:
        title_reducer = reducer.__str__().upper()
    ax.set_title(f"Method: {title_reducer}")
    if coeff is not None and arrows:
        n = coeff.shape[1]
        for i in range(n):
            plt.arrow(0, 0, coeff[0,i]*5, coeff[1,i]*5,color = 'r',alpha = 0.5)
            if labels is None:
                plt.text(coeff[0,i]* 5, coeff[1,i] * 5, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
            else:
                print(labels[i])
                plt.text(coeff[0,i]* 5, coeff[1,i] * 5, labels[i], color = 'g', ha = 'center', va = 'center')
       