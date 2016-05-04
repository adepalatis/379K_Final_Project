from __future__ import division
import pandas as pd
import numpy as np
from sklearn import cross_validation, linear_model, metrics, preprocessing


def save_results(predictions, filename):
    """Given a vector of predictions, save results in CSV format."""
    with open(filename, 'w') as f:
        f.write("id,ACTION\n")
        for i, pred in enumerate(predictions):
            f.write("%d,%f\n" % (i + 1, pred))

def main():
    df = pd.read_csv('../data/train.csv', sep = ',', header = None, skiprows = 1, names = ["ACTION", "RESOURCE", "MGR_ID", "ROLE_ROLLUP_1", "ROLE_ROLLUP_2", "ROLE_DEPTNAME", 
                                                                                "ROLE_TITLE", "ROLE_FAMILY_DESC", "ROLE_FAMILY", "ROLE_CODE"])
    df_test = pd.read_csv('../data/test.csv', sep = ',', header = None, skiprows = 1, names = ["ID", "RESOURCE", "MGR_ID", "ROLE_ROLLUP_1", "ROLE_ROLLUP_2", "ROLE_DEPTNAME", 
                                                                                "ROLE_TITLE", "ROLE_FAMILY_DESC", "ROLE_FAMILY", "ROLE_CODE"])

    dep = df['ACTION']
    indep_vars = ['RESOURCE', 'MGR_ID', 'ROLE_ROLLUP_1', 'ROLE_ROLLUP_2', 'ROLE_DEPTNAME', 'ROLE_TITLE', 
              'ROLE_FAMILY_DESC', 'ROLE_FAMILY', 'ROLE_CODE']
    indep = df[indep_vars]
    x = df_test[indep_vars]
    
    encoder = preprocessing.OneHotEncoder()
    encoder.fit(np.vstack((x, indep)))
    indep = encoder.transform(indep)
    x = encoder.transform(x)

    regr = linear_model.LogisticRegression()
    regr.fit(indep, dep)

    preds = regr.predict(x)

    filename = raw_input("Enter name for submission file: ")
    save_results(preds, filename + ".csv")



if __name__ == "__main__":
    main()