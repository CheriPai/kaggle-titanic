import helpers
import pandas as pd
from sklearn import cross_validation
from sklearn.externals import joblib
from sklearn.svm import SVC


df = pd.read_csv('data/train.csv')
df = helpers.clean_data(df)

# Set up train and cv sets
X = df.drop(['Survived'], axis=1).values
y = df.Survived.values
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    X, y, test_size=0.1, random_state=0)

# Create and train SVM
clf = SVC()
clf.fit(X_train, y_train)

# Test on cross validation set
print('Accuracy: {}'.format(clf.score(X_test, y_test)))

# Export model
joblib.dump(clf, 'data/output/clf.pkl')
