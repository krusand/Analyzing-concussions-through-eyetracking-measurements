from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
from config import *
features = pd.read_parquet(FEATURES_DIR / "features.pq")


y_data = features["y"]
X_data = features.drop(["participant_id", "y"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=.2)

parameters = {'max_depth': [None,1,3,5,7,9,11], 'n_estimators': [10, 100, 1000], 'max_features': ['sqrt', 'log2', None]}

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ('clf', GridSearchCV(RandomForestClassifier(n_jobs=-1), parameters, cv=5, verbose=10))
])

print(pipe.fit(X_train, y_train).score(X_test, y_test))

# results=pd.DataFrame()
# results['columns']=X_train.columns
# results['importances'] = pipe["clf"].feature_importances_
# results.sort_values(by='importances',ascending=False,inplace=True)
print(pipe["clf"].best_params_)
# print(results)
