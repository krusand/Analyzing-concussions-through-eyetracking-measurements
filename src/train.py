from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
from config import *
features = pd.read_parquet(FEATURES_DIR / "features.pq")

sig_features = ['ANTI_SACCADE_total_acceleration_magnitude_right_max',
 'ANTI_SACCADE_total_acceleration_magnitude_right_median',
 'FITTS_LAW_avg_fixations_pr_second',
 'FITTS_LAW_std_fixations_pr_second',
 'FITTS_LAW_mean_amplitude_sacc',
 'FITTS_LAW_mean_duration_sacc',
 'FITTS_LAW_mean_duration_fix',
 'FITTS_LAW_total_acceleration_magnitude_right_mean',
 'FITTS_LAW_total_acceleration_magnitude_right_median',
 'REACTION_reaction_time_avg',
 'REACTION_reaction_time_std',
 'REACTION_total_acceleration_magnitude_right_median',
 'EVIL_BASTARD_mean_duration_sacc',
 'EVIL_BASTARD_total_acceleration_magnitude_right_max',
 'EVIL_BASTARD_total_acceleration_magnitude_right_median',
 'EVIL_BASTARD_distance_to_fixpoint_max',
 'SHAPES_total_acceleration_magnitude_right_mean',
 'SHAPES_total_acceleration_magnitude_right_median',
 'SMOOTH_PURSUITS_mean_peak_velocity_sacc',
 'SMOOTH_PURSUITS_mean_duration_sacc',
 'SMOOTH_PURSUITS_total_acceleration_magnitude_right_max',
 'SMOOTH_PURSUITS_total_acceleration_magnitude_right_median',
 'SMOOTH_PURSUITS_Var_total',
 'SMOOTH_PURSUITS_distance_to_fixpoint_max',
 'SMOOTH_PURSUITS_distance_to_fixpoint_std']

y_data = features["y"]
# X_data = features.drop(["participant_id", "y"], axis=1)
X_data = features[sig_features]

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=.2)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(max_depth=4))
])


print(pipe.fit(X_train, y_train).score(X_test, y_test))

results=pd.DataFrame()
results['columns']=X_train.columns
results['importances'] = pipe["clf"].feature_importances_
results.sort_values(by='importances',ascending=False,inplace=True)

print(results)
