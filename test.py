import helpers
import pandas as pd
from sklearn.externals import joblib


df = pd.read_csv('data/test.csv')
ids = df.PassengerId
df = helpers.clean_data(df)

# Load classifier and run test
clf = joblib.load('data/output/clf.pkl')
predictions = clf.predict(df.values)
predictions = pd.DataFrame({'PassengerId': ids, 'Survived': predictions})
predictions.to_csv('data/output/predictions.csv', index=False)
