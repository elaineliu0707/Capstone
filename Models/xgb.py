# Author: Ben Greenawald

import preprocess as pr
import xgboost as xgb
from sklearn.metrics import f1_score
from datetime import date
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import pandas as pd

"""
base_dir = "/home/benji/Capstone/Models/exported-features/"
group = "Hamas"
features, response = pr.readData(group, base_dir)
print("Train feature shape: " + str(features.shape))
print("Train label length: " + str(len(response)))

### Grid Search not efficiently supported with XGBoost,
# resort to manual parameter search
xg_train = xgb.DMatrix(features, label = response)
del features, response
n_folds = 3
early_stopping = 50
params = {'max_depth': 10, 'nthread':3, 'n_estimators':500,
'learning_rate':0.5, 'objective': 'binary:logistic', 'subsample':0.7}
cv = xgb.cv(params, xg_train, nfold=n_folds, early_stopping_rounds=early_stopping, verbose_eval=1)
print(cv)

Results for group Hamas
Test RMSE for 3 Fold CV, all with early stopping at 10
Depth: 5, Learning Rate:1 , NTrees: 10      0.184657
Depth: 5, Learning Rate:1 , NTrees: 100       0.184657
Depth: 10, Learning Rate:1 , NTrees: 100       0.196157
Depth: 10, Learning Rate:0.5 , NTrees: 100      0.153139
Depth: 10, Learning Rate:0.1 , NTrees: 100       0.24192

Change early stopping to 25
Depth: 10, Learning Rate:0.5 , NTrees: 100      0.153139
Depth: 10, Learning Rate:0.1 , NTrees: 100       0.24192

Change objective to binary logistic
Depth: 10, Learning Rate:0.5 , NTrees: 100      0.025562
Depth: 10, Learning Rate:0.1 , NTrees: 100      0.039132

Change subsample to 0.7
Depth: 10, Learning Rate:0.5 , NTrees: 100      0.026114
"""
# From the results, we see that depth is much more important
# than the number of estimators.

# Load in the data
base_dir = "/home/benji/Documents/capstone/Models/exported-features/"
for group in pr.groups:
    features, response = pr.readData(group, base_dir)
    print("Train feature shape: " + str(features.shape))
    print("Train label length: " + str(len(response)))

    ros = RandomOverSampler(random_state=0)
    features, response = ros.fit_sample(features, response)
    print(sorted(Counter(response).items()))

    features = pd.DataFrame(features)
    response = np.array(response)

    xg_train = xgb.DMatrix(features, label = response)
    params = {'max_depth': 10, 'nthread':6, 'n_estimators':500,
    'learning_rate':0.5, 'objective': 'binary:logistic'}
    bst = xgb.train(params, xg_train, verbose_eval = 0)

    # Read in the test data
    test_features, test_response = pr.readData(group, base_dir, train=False, colLen = features.shape[1])
    print("Test feature shape: " + str(test_features.shape))
    print("Test label length: " + str(len(test_response)))

    transformed_features = xgb.DMatrix(test_features)

    preds1 = bst.predict(transformed_features)
    preds = np.array(preds1)
    preds[preds >= 0.5] = 1
    preds[preds < 0.5] = 0

    print(preds)
    print(sum(preds == test_response)/len(preds))
    print(f1_score(test_response, preds, pos_label=test_response[0]))
    # Evaluate the results
    with open("/home/benji/Documents/capstone/Results/XGBoost/results-{0}.txt".format(str(date.today())), "a+") as file:
        file.write(group + "\n")
        file.write("Accurary: " + str(sum(preds == test_response)/len(preds)) + "\n")
        file.write("F1-Score: " + str(f1_score(test_response, preds, pos_label=test_response[0])) + "\n\n")
        file.close()
