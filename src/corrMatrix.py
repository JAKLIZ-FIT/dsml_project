import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("manufacturing.csv")
le = LabelEncoder()
data['ionizationclas'] = le.fit_transform(data['ionizationclass'])
data['FluxCompensation'] = le.fit_transform(data['FluxCompensation'])
data['multideminsionality']= le.fit_transform(data['multideminsionality'])
data['error'] = le.fit_transform(data['error'])
data['error_type'] = le.fit_transform(data['error_type'])

data.drop("id", axis = 1,inplace=True)

corrMatrix = data.corr()
print(corrMatrix)

