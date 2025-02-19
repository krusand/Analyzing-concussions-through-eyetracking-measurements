import pandas as pd

data = pd.read_excel("Data/survey_responses.xlsx", sheet_name="all")

print(data.shape)

# Drop rows with NA values
data = data.dropna(axis=0)

# Drop columns with NA values
data = data.dropna(axis=1)

