from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
from config import *
features = pd.read_parquet(FEATURES_DIR / "anti_saccade_features.pq")

y_data = features["y"]
X_data = features.drop(["experiment", "participant_id", "y"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=.2)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(max_depth=2))
])


print(pipe.fit(X_train, y_train).score(X_test, y_test))

results=pd.DataFrame()
results['columns']=X_train.columns
results['importances'] = pipe["clf"].feature_importances_
results.sort_values(by='importances',ascending=False,inplace=True)

print(results)
