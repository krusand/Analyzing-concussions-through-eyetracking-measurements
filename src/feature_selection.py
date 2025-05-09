import pandas as pd
import numpy as np

from functools import reduce
import argparse

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from kneed import KneeLocator
from scipy import stats
from statsmodels.stats.multitest import multipletests

from config import *

def load_features(experiments: list[str]) -> pd.DataFrame:
    logging.info("Loading features")
    df_features_list = []
    
    for experiment in experiments:
        df = pd.read_parquet(FEATURES_DIR / f"{experiment}_features.pq")
        df.columns = [f'{experiment}_{column}' if column not in ['experiment', 'participant_id'] else f'{column}' for column in df.columns]
        df = df.drop("experiment", axis=1)
        df_features_list.append(df)
    
    df_features = reduce(lambda x, y: pd.merge(x, y, on = ["participant_id"], how="outer"), df_features_list)
    
    logging.info("Finished loading features")
    
    return df_features

def load_demographic_info() -> pd.DataFrame:
    logging.info("Loading demographics")
    demographics = pd.read_excel(DATA_DIR / "demographic_info.xlsx")
    
    # Filter
    demographics = demographics[demographics["Eye tracking date"].notna()]
    
    # Mutate
    demographics["y"] = (demographics["Group"] == "PATIENT").astype(int)
    demographics["participant_id"] = demographics["ID"].astype(int)
    
    # Select
    demographics = demographics[["participant_id", "y"]]
    return demographics

def join_features_on_demographic_info(feature_df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Joining features on demographics")
    demographics = load_demographic_info()
    return pd.merge(demographics, feature_df, how='left', on='participant_id')

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
    # Apply variance threshold
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)

    # Get selected feature names
    selected_feature_names = X.columns[selector.get_support()]

    # Create reduced DataFrame
    X_reduced = X[selected_feature_names]

    logging.info(f"Kept {X_reduced.shape[1]} features out of {X.shape[1]}")
    
    return X_reduced

def remove_multicollineraity(X, y, feature_corr_threshold=0.85):
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
    

def get_significant_features(X, y, alpha=0.05, return_details=False):
    """
    Identify features that show statistically significant differences between two groups.
    
    Parameters:
    -----------
    X : DataFrame
        Feature matrix
    y : Series or DataFrame
        Binary target variable (0/1 or similar)
    alpha : float, optional (default=0.05)
        Significance level threshold
        
    Returns:
    --------
    significant_features : list or DataFrame
        If return_details=False, returns list of feature names
        If return_details=True, returns DataFrame with features, p-values, and test statistics
    """
    # Convert y to series
    y = y.iloc[:, 0]
    
    # Merge data for filtering
    data = X.copy()
    data['y'] = y
    
    control = data[data['y'] == 0]
    patient = data[data['y'] == 1]
    
    test_statistics = []

    for feature in X.columns:
        
        if feature in ["participant_id", 'y']:
            continue
        
        control_values = control[control[feature].notna()][feature]
        patient_values = patient[patient[feature].notna()][feature]
        
        U_value, p_value = stats.mannwhitneyu(control_values, patient_values)
        
        # Save test statistics
        test_statistics.append({
            'feature': feature,
            'p_value': p_value,
            'U_value': U_value,
            'mean_control': float(np.nanmean(control_values)),
            'mean_patient': float(np.nanmean(patient_values)),
            'std_control': float(np.nanstd(control_values)),
            'std_patient': float(np.nanstd(patient_values))
        })
    
    # Convert to DataFrame
    test_statistics = pd.DataFrame(test_statistics)
    if len(test_statistics) > 0:
        test_statistics = test_statistics.sort_values('p_value')         

    # Get significant features
    reject, p_values_corrected, _, _ = multipletests(test_statistics["p_value"], alpha=alpha, method='fdr_bh')
    test_statistics["p_value_corrected"] = p_values_corrected
    significant_features = test_statistics[reject]

    logging.info(f"Found {len(significant_features)} significant features.")

    if not return_details:
        significant_features = significant_features["feature"]

    return significant_features

def get_feature_importance(X, y, by_experiment, cv=5, n_jobs=-1, random_state=42):
    """
    Calculate feature importance for each experiment by training optimized models on experiment-specific features.
    
    Parameters:
    -----------
    X : DataFrame
        Feature matrix
    y : Series or DataFrame
        Target variable
    experiments : list, optional (default=None)
        List of experiment prefixes to group features by. If None, will try to infer from column names.
    cv : int, optional (default=5)
        Number of cross-validation folds
    n_jobs : int, optional (default=-1)
        Number of parallel jobs for GridSearch and RandomForest
    random_state : int, optional (default=42)
        Random seed for reproducibility
        
    Returns:
    --------
    feature_importances : DataFrame
        DataFrame with feature importances for each feature
    best_params : dict
        Dictionary mapping experiment names to their best parameters
    """
    if by_experiment:
        experiments = ["ANTI_SACCADE", "FITTS_LAW", "FIXATIONS", "KING_DEVICK", "EVIL_BASTARD", "REACTION", "SHAPES", "SMOOTH_PURSUITS"]
    else:
        experiments = ["ALL"]

    # Convert y to correct format
    y = y.values.ravel()

    # Initialize results
    feature_importances = pd.DataFrame(
        np.zeros(X.shape[1]), 
        columns=["Importance"], 
        index=X.columns)
    best_params = {}
    
    for exp in experiments:
        if exp == "ALL":
            exp_features = X.columns
            X_exp = X
        else:
            exp_features = [f for f in X.columns if exp in f]
            X_exp = X[exp_features]
        
        # Hyperparameters for GridSearch
        parameters = {
            'max_depth': [None, 3, 5, 7, 9, 11],
            'n_estimators': [20, 30, 40, 50, 70, 100],
            'max_features': ['sqrt', 'log2', None]
        }

        # Find best clf
        grid_search = GridSearchCV(
            RandomForestClassifier(n_jobs=n_jobs, random_state=random_state), 
            parameters, 
            cv=cv, 
            verbose=0, 
            n_jobs=n_jobs)
        grid_search.fit(X_exp, y)
        
        best_model = grid_search.best_estimator_
        importances = best_model.feature_importances_
        best_params[exp] = grid_search.best_params_
        
        # update importances
        feature_importances.loc[exp_features, "Importance"] += importances
    
    return feature_importances, best_params

def get_important_features(feature_importances, by_experiment, curve='convex', 
                    direction='decreasing', min_features=3, max_features=None):
    """
    Select important features for each experiment using the knee/elbow method on feature importance values.
    
    Parameters:
    -----------
    feature_importances : DataFrame
        DataFrame with feature importances, index=feature names, columns=['Importance']
    experiments : list, optional (default=None)
        List of experiment prefixes to group features by. If None, will try to infer from feature names.
    curve : str, optional (default='convex')
        Type of curve to fit for knee detection ('convex' or 'concave')
    direction : str, optional (default='decreasing')
        Direction of values ('decreasing' or 'increasing')
    min_features : int, optional (default=3)
        Minimum number of features to select even if knee suggests fewer
    max_features : int, optional (default=None) 
        Maximum number of features to select regardless of knee
        
    Returns:
    --------
    important_features : dict
        Dictionary mapping experiment names to DataFrames of selected features with their importances
    summary : DataFrame
        DataFrame summarizing the number of features selected for each experiment
    """
    if by_experiment:
        experiments = ["ANTI_SACCADE", "FITTS_LAW", "FIXATIONS", "KING_DEVICK", "EVIL_BASTARD", "REACTION", "SHAPES", "SMOOTH_PURSUITS"]
    else:
        experiments = ["ALL"]
    
    # Store results
    selected_features = {}
    summary_rows = []
    
    for exp in experiments:
        # Get features for this experiment
        if exp == "ALL":
            feature_importances_exp = feature_importances
        else:
            feature_importances_exp = feature_importances.filter(like=exp,axis=0)
        
        # Sort features by importance
        sorted_features = feature_importances_exp.sort_values(by="Importance", ascending=False)
        importance_values = sorted_features["Importance"].values
        feature_ranks = np.arange(len(importance_values))

        # Find knee
        knee = KneeLocator(
            feature_ranks, 
            importance_values, 
            curve=curve, 
            direction=direction
        )
        knee_point = knee.knee

        if knee_point is not None:
            logging.info(f"{exp}: Knee detected at feature rank {knee_point}")
        else:
            logging.warning("Knee point is not detected")
            knee_point = len(feature_importances_exp) - 1

        # Select features above knee
        selected_features_exp = sorted_features.iloc[:knee_point+1]
        logging.info(f"Automatically selected {selected_features_exp.shape[0]} features")
        
        selected_features[exp]=selected_features_exp
        
        # Add summary data
        summary_rows.append({
            'Experiment': exp,
            'Total Features': len(feature_importances_exp),
            'Selected Features': len(selected_features_exp),
            'Selection %': round(len(selected_features_exp) / len(feature_importances_exp) * 100, 1),
            'Top Feature': selected_features_exp.index[0] if not selected_features_exp.empty else None,
            'Top Importance': selected_features_exp['Importance'].iloc[0] if not selected_features_exp.empty else None,
            'Method': "kneedle"
        })
         
    # Create summary dataframe
    summary = pd.DataFrame(summary_rows)
    
    # Concatenate all important features
    selected_features = pd.concat(selected_features.values(), keys=selected_features.keys()).reset_index(level=0, drop=True)
        
    return selected_features, summary

def select_features(features: pd.DataFrame) -> None:
    
    # Split into x and y
    y = pd.DataFrame(features["y"], columns=["y"])
    X = features.drop(["participant_id", "y"], axis=1)
    
    X_reduced = (X
        .pipe(remove_low_variance_features)
        .pipe(remove_multicollineraity, y)
        .pipe(remove_outliers, 'MAD', 10.0)
    )
    
    significant_features = get_significant_features(X_reduced, y, return_details=False)
    
    # Get important features by experiment
    feature_importances_exp, _ = get_feature_importance(X_reduced, y, by_experiment=True)
    important_features_exp, _ = get_important_features(feature_importances_exp, by_experiment=True)
    logging.info("Successfully found important features by experiment")

    # Get important features overall
    feature_importances, _ = get_feature_importance(X_reduced, y, by_experiment=False)
    important_features, _ = get_important_features(feature_importances, by_experiment=False)
    logging.info("Successfully found important features overall")
    
    # Get union of selected features
    feature_list = list(set(important_features.index).union(set(important_features_exp.index)).union(set(significant_features)))
    
    # Save dataframe with selected features
    selected_features = features[["participant_id", "y"] + feature_list]
    
    return selected_features

def main(args: argparse.ArgumentParser) -> None:
    logging.info("Load features and join with demographic info")
    print(args.experiments)
    features = load_features(args.experiments)
    data = join_features_on_demographic_info(feature_df=features)
    data.to_parquet(FEATURES_DIR / 'all_features.pq')
    # Select features
    selected_features = select_features(data)
    logging.info("Successfully selected features")
    
    # Save to parquet
    selected_features.to_parquet(FEATURES_DIR / 'features.pq')
    logging.info("Saved features to file")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run feature extraction")
    parser.add_argument("--experiments", nargs='+', required=False, help="List of experiment names")
    parser.add_argument('--all_experiments', required=False, action=argparse.BooleanOptionalAction, help="Run pipeline for all experiments")
    args = parser.parse_args() 
    
    if (not args.all_experiments or args.all_experiments is None) and (args.experiments is None):
        parser.error("--experiments must be provided when not running all experiments")

    if args.all_experiments:
        args.experiments = ["ANTI_SACCADE", "FITTS_LAW", "FIXATIONS", "KING_DEVICK", "EVIL_BASTARD", "REACTION", "SHAPES", "SMOOTH_PURSUITS"]
    
    main(args)