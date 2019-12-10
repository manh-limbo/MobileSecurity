import pandas as pd
import numpy as np
import pickle
from sklearn.svm import SVC
import sklearn.ensemble as ske
from sklearn import tree, linear_model
from sklearn.feature_selection import SelectFromModel
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.utils import class_weight
# import xgboost

data = pd.read_csv('data/out_features.txt', sep=';')
X = data.drop(['Package', 'Class'], axis=1).values
y = data['Class'].values
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train1 = X_train[:, 3:]
X_test1 = X_test[:, 3:]
X_train2 = X_train[:, 3:28]
X_test2 = X_test[:, 3:28]
X_train3 = X_train[:, 28:]
X_test3 = X_test[:, 28:]

np.savetxt('classifier/x_train.txt', X_train, delimiter=';')
np.savetxt('classifier/y_train.txt', y_train, delimiter=';')
np.savetxt('classifier/x_test.txt', X_test, delimiter=';')
np.savetxt('classifier/y_test.txt', y_test, delimiter=';')



#Algorithm comparison
algorithms = {
        "GaussianNB": GaussianNB(),
        "DecisionTree": tree.DecisionTreeClassifier(max_depth=15),
        "RandomForest": ske.RandomForestClassifier(n_estimators=51),
        "GradientBoosting": ske.GradientBoostingClassifier(n_estimators=31),
        "AdaBoost": ske.AdaBoostClassifier(n_estimators=31)
    }

results = {}
print("\nNow testing algorithms")
for algo in algorithms:
    clf = algorithms[algo]

    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print("%s : %f %%" % (algo, score*100))
    res = clf.predict(X_test)
    mt = confusion_matrix(y_test, res)
    print("False positive rate ALL: %f %%" % ((mt[0][1] / float(sum(mt[0]))) * 100))
    print('False negative rate ALL: %f %%' % ((mt[1][0] / float(sum(mt[1])) * 100)))
    joblib.dump(clf, 'classifier/classifier_' + 'ALL_' + algo + '.pkl')

    clf.fit(X_train1, y_train)
    score = clf.score(X_test1, y_test)
    print("%s : %f %%" % (algo, score*100))
    res = clf.predict(X_test1)
    mt = confusion_matrix(y_test, res)
    print("False positive rate 12: %f %%" % ((mt[0][1] / float(sum(mt[0]))) * 100))
    print('False negative rate 12: %f %%' % ((mt[1][0] / float(sum(mt[1])) * 100)))
    joblib.dump(clf, 'classifier/classifier_' + '12_' + algo + '.pkl')

    clf.fit(X_train2, y_train)
    score = clf.score(X_test2, y_test)
    print("%s : %f %%" % (algo, score*100))
    res = clf.predict(X_test2)
    mt = confusion_matrix(y_test, res)
    print("False positive rate 1: %f %%" % ((mt[0][1] / float(sum(mt[0]))) * 100))
    print('False negative rate 1: %f %%' % ((mt[1][0] / float(sum(mt[1])) * 100)))
    joblib.dump(clf, 'classifier/classifier_' + '1_' + algo + '.pkl')

    clf.fit(X_train3, y_train)
    score = clf.score(X_test3, y_test)
    print("%s : %f %%" % (algo, score*100))
    res = clf.predict(X_test3)
    mt = confusion_matrix(y_test, res)
    print("False positive rate 2: %f %%" % ((mt[0][1] / float(sum(mt[0]))) * 100))
    print('False negative rate 2: %f %%' % ((mt[1][0] / float(sum(mt[1])) * 100)))
    joblib.dump(clf, 'classifier/classifier_' + '2_' + algo + '.pkl')
