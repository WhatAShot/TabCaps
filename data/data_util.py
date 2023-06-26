import pandas as pd
from sklearn.datasets import load_svmlight_file
from glob import glob
import os
def svm2pkl(source, save_path):
    X_train, y_train = load_svmlight_file(file)
    # X_valid, y_valid = load_svmlight_file(os.path.join(source, 'vali'))
    # X_test, y_test = load_svmlight_file(os.path.join(source, 'test'))

    X_train = pd.DataFrame(X_train.todense())
    y_train = pd.Series(y_train)
    return pd.concat([y_train, X_train], axis=1).T.reset_index(drop=True).T

    X_valid = pd.DataFrame(X_valid.todense())
    y_valid = pd.Series(y_valid)
    pd.concat([y_valid, X_valid], axis=1).T.reset_index(drop=True).T.to_pickle(os.path.join(save_path, 'valid.pkl'))

    X_test = pd.DataFrame(X_test.todense())
    y_test = pd.Series(y_test)
    pd.concat([y_test, X_test], axis=1).T.reset_index(drop=True).T.to_pickle(os.path.join(save_path, 'test.pkl'))

def csv2pkl(source, save_path):
    data = pd.read_csv(source)
    data.to_pickle(save_path)

if __name__ == '__main__':
    source = '/data2/liaokuanlun/dataset/tabular/click.csv'
    csv2pkl(source, '/data2/liaokuanlun/dataset/tabular/click.pkl')