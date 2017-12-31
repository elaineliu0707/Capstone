import preprocess as pr
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from datetime import date

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
base_dir = "C:/Users/Ben/Documents/Capstone/Models/exported-features/"
for group in pr.groups:
    features, response = pr.readData(group, base_dir)
    print("Train feature shape: " + str(features.shape))
    print("Train label length: " + str(len(response)))

    # Make classifier using the best parameters, increase depth
    rf = RandomForestClassifier(random_state=0, max_depth=8,
        n_estimators=1000, n_jobs=4)
    rf.fit(features, response)
    print("Building Classifier")
    rf.fit(features, response)

    # Read in the test data
    test_features, test_response = pr.readData(group, base_dir, train=False, colLen = features.shape[1])
    print("Test feature shape: " + str(test_features.shape))
    print("Test label length: " + str(len(test_response)))
    preds = rf.predict(test_features)

    # Evaluate the results
    with open("C:/Users/Ben/Documents/Capstone/Results/RandomForest/results-{0}.txt".format(str(date.today())), "a+") as file:
        file.write("{0}, ".format(group) + str(f1_score(test_response, preds, pos_label=test_response[0])) + "\n")
        file.close()
