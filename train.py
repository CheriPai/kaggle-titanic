import helpers
import pandas as pd
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import average_precision_score, recall_score


df = pd.read_csv('data/train.csv')
df = helpers.clean_data(df)

# Set up train and cv sets
X = df.drop(['Survived'], axis=1).values
y = df.Survived.values
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    X, y, test_size=0.1, random_state=0)

# Create and train classifier
clf = RandomForestClassifier(random_state=1, n_estimators=100, min_samples_split=4, min_samples_leaf=2)
clf.fit(X_train, y_train)

# Test on cross validation set
y_predict = clf.predict(X_test)
print('Accuracy:  %.4f' % clf.score(X_test, y_test))
print('Precision: %.4f' % average_precision_score(y_test, y_predict))
print('Recall:    %.4f' % recall_score(y_test, y_predict))

# Export model
joblib.dump(clf, 'data/output/clf.pkl')
