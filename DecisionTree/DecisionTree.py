import numpy as np
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext, SparkConf
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel

def load_data(filename, use_labels=True):
    """
    Load data from CSV files and return them as numpy arrays
    The use_labels parameter indicates whether one should
    read the first column (containing class labels). If false,
    return all 0s.
    """

    # load column 1 to 8 (ignore last one)
    data = np.loadtxt(open("data/" + filename), delimiter=',',
                      usecols=range(1, 9), skiprows=1)
    if use_labels:
        labels = np.loadtxt(open("data/" + filename), delimiter=',',
                            usecols=[0], skiprows=1)
    else:
        labels = np.zeros(data.shape[0])
    return labels, data


# Append the label value to the end of the list of features for each user
def reformatData(features, labels):
    data = []
    for x in range(len(labels)):
        data.append(LabeledPoint(labels[x], features[x]))
    return data


def testDecisionTree(model, testData):
    predictions = model.predict(testData.map(lambda x: x.features))

    # Save the output in a csv file
    filename = 'DecisionTree/decision_tree_results'
    save_results(predictions, filename + ".csv")


def save_results(predictions, filename):
    """Given a vector of predictions, save results in CSV format."""
    predsList = predictions.take(predictions.count())
    with open(filename, 'w') as f:
        f.write("id,ACTION\n")
        for i, pred in enumerate(predsList):
            f.write("%d,%f\n" % (i + 1, pred))


def main():
    sc = SparkContext(appName="MyApp")
    sc.setLogLevel('ERROR')

    # Parse data
    train_labels, train_data = load_data('train.csv')
    dummy_labels, test_data = load_data('test.csv', use_labels=False)

    # Map each data point's label to its features
    train_set = reformatData(train_data, train_labels)
    test_set = reformatData(test_data, dummy_labels)

    # Parallelize the data
    parallelized_train_set = sc.parallelize(train_set)
    parallelized_test_set = sc.parallelize(test_set)

    # Split the data
    trainSet, validationSet = parallelized_train_set.randomSplit([1.0, 0.0], seed=42)

    # Train the models
    decisionTreeModel = DecisionTree.trainClassifier(trainSet, numClasses=5, categoricalFeaturesInfo={},
                                         impurity='gini', maxBins=55, maxDepth=30, minInstancesPerNode=2)

    # Test the model
    testDecisionTree(decisionTreeModel, parallelized_test_set)

if __name__ == '__main__':
    main()