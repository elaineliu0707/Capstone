# Author: Ben Greenawald

import preprocess as pr
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from datetime import date

# Find best parameters using Hamas
"""
base_dir = "/home/benji/Capstone/Models/exported-features/"
group = "Hamas"
features, response = pr.readData(group, base_dir)
print("Train feature shape: " + str(features.shape))
print("Train label length: " + str(len(response)))

### Hyperparameter optimization using grid search
svc = SVC()
params = {'C': [100,10000], 'kernel':['rbf']}
clf = GridSearchCV(svc, params, verbose=5, n_jobs=3)
clf.fit(features, response)
print(clf.best_params_)

Results
0.884 (+/-0.275) for {'kernel': 'rbf', 'C': 100}
0.884 (+/-0.275) for {'kernel': 'rbf', 'C': 1000}
0.884 (+/-0.275) for {'kernel': 'rbf', 'C': 10000}
0.892 (+/-0.255) for {'kernel': 'sigmoid', 'C': 100}
0.892 (+/-0.255) for {'kernel': 'sigmoid', 'C': 1000}
0.892 (+/-0.255) for {'kernel': 'sigmoud', 'C': 10000}


pr.evaluateGridSearch(clf)
"""

# From the results we see that the result are pretty much
# the same across the board, but sigmoud seems slightly
# better

# Load in the data
base_dir = "/home/benji/Capstone/Models/exported-features/"
for group in pr.groups:
    features, response = pr.readData(group, base_dir)
    print("Train feature shape: " + str(features.shape))
    print("Train label length: " + str(len(response)))

    # Make classifier using the best parameters, increase depth
    rf = RandomForestClassifier(random_state=0, max_depth=6,
        n_estimators=2000, n_jobs=3)
    rf.fit(features, response)
    print("Building Classifier")
    rf.fit(features, response)

    # Read in the test data
    test_features, test_response = pr.readData(group, base_dir, train=False, colLen = features.shape[1])
    print("Test feature shape: " + str(test_features.shape))
    print("Test label length: " + str(len(test_response)))
    preds = rf.predict(test_features)

    print(preds)
    print(sum(preds == test_response)/len(preds))
    print(f1_score(test_response, preds, pos_label=test_response[0]))
    # Evaluate the results
    with open("/home/benji/Capstone/Results/SVM/results-{0}.txt".format(str(date.today())), "a+") as file:
        file.write("{0}, ".format(group) + str(f1_score(test_response, preds, pos_label=test_response[0])) + "\n")
        file.close()"""
