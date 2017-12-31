# Author: Ben Greenawald

# Translating the Python code and implementing Random Forest and XGBoost

# Import Statement

import pandas as pd
import scipy.sparse as sp
import numpy as np

def readData(groupName, base_dir, train=True, colLen = None):

    if train:
        print("Reading in train features for " + groupName)
        filename = "{0}/{1}TrainFeatures.txt".format(groupName, groupName)
    else:
        print("Reading in test features for " + groupName)
        filename = "{0}/{1}TestFeatures.txt".format(groupName, groupName)

    with open(base_dir + filename, "r") as Features:
        rows = np.array([int(x.strip()) for x in Features.readline().strip().split(",")])
        cols = np.array([int(x.strip()) for x in Features.readline().strip().split(",")])
        vals = np.array([float(x.strip()) for x in Features.readline().strip().split(", ")])
        Features.close()


    row_len = max(rows) + 1

    if colLen:
        col_len = colLen
    else:
        col_len = max(cols) + 1

    features = pd.DataFrame(sp.coo_matrix((vals, (rows, cols)), shape=(row_len, col_len)).toarray())
    del rows, cols, vals

    if train:
        print("Reading in train labels for " + groupName)
        filename = "{0}/{1}TrainLabels.txt".format(groupName, groupName)
    else:
        print("Reading in test labels for " + groupName)
        filename = "{0}/{1}TestLabels.txt".format(groupName, groupName)

    # Read in the labels
    with open(base_dir + filename, "r") as Labels:
        labels = np.array([int(float(x.strip())) for x in Labels.readline().strip().split(", ")])
        Labels.close()

    return((features, labels))

def evaluateGridSearch(clf, y_test):
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

# List of group names
groups = [
        "Aljazeera",
        "CNN",
        "Mohamed Rateb Al-Nabulsi",
        "Movement of Society for Peace",
        "Tunisian General Union of Labor",
        "Rabee al-Madkhali",
        "Socialist Union Morocco",
        "Salman Fahd Al-Ohda",
        "Alarabiya",
        "GA on Islamic Affairs",
        "Al Shabaab",
        "Ansar Al Sharia",
        "AQIM",
        "Azawad",
        "ISIS",
        "Syrian Democratic Forces",
        "Houthis",
        "Hezbollah",
        "Hamas",
        "Al-Boraq"
]