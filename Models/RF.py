# Author: Ben Greenawald

import preprocess as pr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from datetime import date

# Group used: Hamas
### Hyperparameter optimization using grid search
#rf = RandomForestClassifier(random_state=0)
#params = {'max_depth': [2,4,6], 'n_estimators':[500, 1000, 3000]}
#clf = GridSearchCV(rf, params)
#clf.fit(features, response)
#print(clf.best_params_)
"""
Results
0.867 (+/-0.229) for {'max_depth': 2, 'n_estimators': 500}
0.866 (+/-0.227) for {'max_depth': 2, 'n_estimators': 1000}
0.867 (+/-0.229) for {'max_depth': 2, 'n_estimators': 3000}
0.946 (+/-0.115) for {'max_depth': 4, 'n_estimators': 500}
0.946 (+/-0.114) for {'max_depth': 4, 'n_estimators': 1000}
0.946 (+/-0.113) for {'max_depth': 4, 'n_estimators': 3000}
0.951 (+/-0.110) for {'max_depth': 6, 'n_estimators': 500}
0.951 (+/-0.110) for {'max_depth': 6, 'n_estimators': 1000}
0.951 (+/-0.110) for {'max_depth': 6, 'n_estimators': 3000}
"""
# From the results, we see that depth is much more important
# than the number of estimators.

# pr.evaluateGridSearch(clf)

# Load in the data
base_dir = "/home/benji/Capstone/Models/exported-features/"
for group in pr.groups:
    features, response = pr.readData(group, base_dir)
    print("Train feature shape: " + str(features.shape))
    print("Train label length: " + str(len(response)))

    # Make classifier using the best parameters, increase depth
    rf = RandomForestClassifier(random_state=0, max_depth=8,
        n_estimators=3000, n_jobs=3)
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
    with open("/home/benji/Capstone/Results/RandomForest/results-{0}.txt".format(str(date.today())), "a+") as file:
        file.write(group + "\n")
        file.write("Accurary: " + str(sum(preds == test_response)/len(preds)) + "\n")
        file.write("F1-Score: " + str(f1_score(test_response, preds, pos_label=test_response[0])) + "\n\n")
        file.close()
