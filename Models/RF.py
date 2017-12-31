# Load in the data
base_dir = "C:/Users/Ben/Documents/Capstone/Models/exported-features/"
group = "CNN"
features, response = readData(group, base_dir)
print("Train feature shape: " + str(features.shape))
print("Train label length: " + str(len(response)))

# Build the classifier
print("Building Classifier")
clf = RandomForestClassifier(max_depth=4, random_state=0, n_estimators=1000)
clf.fit(features, response)

# Read in the test data
test_features, test_response = readData(group, base_dir, train=False, colLen = features.shape[1])
print("Test feature shape: " + str(test_features.shape))
print("Test label length: " + str(len(test_response)))
preds = clf.predict(test_features)
print(preds)

# Evaluate the results
print(sum(preds == test_response)/len(preds))