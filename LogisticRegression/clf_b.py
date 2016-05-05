from __future__ import division
import pandas as pd
import numpy as np
from sklearn import cross_validation, linear_model, metrics, preprocessing

SEED = 42

def save_results(predictions, filename):
    """Given a vector of predictions, save results in CSV format."""
    #this is borrowed from one of the starting codes
    with open(filename, 'w') as f:
        f.write("id,ACTION\n")
        for i, pred in enumerate(predictions):
            f.write("%d,%f\n" % (i + 1, pred))

def loadData(isTrain):
    """Load data from test and train files"""
    fileRoute = ''
    if isTrain:
        fileRoute = '../data/train.csv'
    else:
        fileRoute = '../data/test.csv'

    df = pd.read_csv(fileRoute, sep = ',', header = None, skiprows = 1)
    indep_vars = [1,2,3,4,5,6,7,8,9]

    if isTrain:
        y = df[0]
        x = df[indep_vars]
    else:
        y = np.zeros(df.shape[0])
        x = df[indep_vars]

    return x, y


def main():
    """ Fit and predict
        Used one-hot encoding from sklearn.preprocessing
        Tried standardscale

    """
    # === Make the model with logistic regressor
    model = linear_model.LogisticRegression(C = 3.0)

    # === Loading data from csv files === #
    print "Loading data"
    indep, dep = loadData(True)
    x, y = loadData(False)

    # === encode all the data with one hot encoder === #
    #encode all categorical ID on both test and train set
    encoder = preprocessing.OneHotEncoder()
    encoder.fit(np.vstack((x, indep)))
    indep = encoder.transform(indep)
    x = encoder.transform(x)

    # === training and see mean == #
    # this part is borrow from one off the starting code
    mean_auc = 0.0
    n = 10  # repeat the CV procedure 10 times to get more precise results
    for i in range(n):
        # for each iteration, randomly hold out 20% of the data as CV set
        indep_train, indep_cv, dep_train, dep_cv = cross_validation.train_test_split(
            indep, dep, test_size=.20, random_state=i*SEED)

        # if you want to perform feature selection / hyperparameter
        # optimization, this is where you want to do it

        # train model and make predictions
        model.fit(indep_train, dep_train) 
        preds = model.predict_proba(indep_cv)[:, 1]

        # compute AUC metric for this CV fold
        fpr, tpr, thresholds = metrics.roc_curve(dep_cv, preds)
        roc_auc = metrics.auc(fpr, tpr)
        print "AUC (fold %d/%d): %f" % (i + 1, n, roc_auc)
        mean_auc += roc_auc

    print "Mean AUC: %f" % (mean_auc/n)

    # === Prediction and output === #
    #fit a nd predict
    model.fit(indep, dep)
    preds = model.predict_proba(x)[:, 1]

    filename = raw_input("Enter name for submission file: ")
    save_results(preds, filename + ".csv")



if __name__ == "__main__":
    main()