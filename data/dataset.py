import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from category_encoders import LeaveOneOutEncoder
from sklearn.impute import MissingIndicator
from sklearn.model_selection import train_test_split
import random
random.seed(1)
def make_split_idx(data):
    train_valid, test = train_test_split(data, test_size=0.2, random_state=88, shuffle=True)
    train, valid = train_test_split(train_valid, test_size=0.15, random_state=88, shuffle=True)
    train_idx = pd.DataFrame(train.index)
    valid_idx = pd.DataFrame(valid.index)
    test_idx = pd.DataFrame(test.index)
    train_idx.to_pickle('./data/hill/train_idx.pkl')
    valid_idx.to_pickle('./data/hill/valid_idx.pkl')
    test_idx.to_pickle('./data/hill/test_idx.pkl')

def remove_unused_column(data):
    unused_list = []
    for col in data.columns:
        uni = len(data[col].unique())
        if uni <= 1:
            unused_list.append(col)
    data.drop(columns=unused_list, inplace=True)
    return data

def quantile_transform(X_train, X_valid, X_test):
    quantile_train = np.copy(X_train)
    qt = QuantileTransformer(random_state=1330, output_distribution='normal').fit(quantile_train)
    X_train = qt.transform(X_train)
    X_valid = qt.transform(X_valid)
    X_test = qt.transform(X_test)
    return X_train, X_valid, X_test

def standard_normalized(X_train, X_valid, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = (X_train - mean) / std
    X_valid = (X_valid - mean) / std
    X_test = (X_test - mean) / std
    return X_train, X_valid, X_test

class Dataset():
    def __init__(self, sub):
        self.sub = sub

    def epsilon(self):
        target = 0
        X_train = np.load('./data/epsilon/N_train.npy')
        X_valid = np.load('./data/epsilon/N_val.npy')
        X_test = np.load('./data/epsilon/N_test.npy')
        y_train = np.load('./data/epsilon/y_train.npy')
        y_valid = np.load('./data/epsilon/y_val.npy')
        y_test = np.load('./data/epsilon/y_test.npy')

        X_train, X_valid, X_test = standard_normalized(X_train, X_valid, X_test)
        return X_train, y_train, X_valid, y_valid, X_test, y_test

    def click(self):
        target = 'target'
        data = pd.read_pickle('./data/click/click.pkl')
        if self.sub == True:
            choose_train_idx = pd.read_pickle('./data/click/choose_train_idx.pkl')[0].values
            train_valid = data.iloc[choose_train_idx, :]
            X_test = data.drop(choose_train_idx)
            X_train = train_valid
            X_valid = X_test.copy()
        else:
            train_idx = pd.read_pickle('./data/click/train_idx.pkl')[0].values
            valid_idx = pd.read_pickle('./data/click/valid_idx.pkl')[0].values
            X_train_valid, X_test = data[:-100_000], data[-100_000:]
            X_train = X_train_valid.iloc[train_idx, :]
            X_valid = X_train_valid.iloc[valid_idx, :]

        y_train, y_valid, y_test = X_train[target].values, X_valid[target].values, X_test[target].values
        X_train = X_train.drop([target], axis=1)
        X_valid = X_valid.drop([target], axis=1)
        X_test = X_test.drop([target], axis=1)

        cat_features = ['url_hash', 'ad_id', 'advertiser_id', 'query_id',
                        'keyword_id', 'title_id', 'description_id', 'user_id']

        cat_encoder = LeaveOneOutEncoder()
        cat_encoder.fit(X_train[cat_features], y_train)
        X_train[cat_features] = cat_encoder.transform(X_train[cat_features])
        X_valid[cat_features] = cat_encoder.transform(X_valid[cat_features])
        X_test[cat_features] = cat_encoder.transform(X_test[cat_features])

        X_train, X_valid, X_test = quantile_transform(X_train.astype(np.float32), X_valid.astype(np.float32), X_test.astype(np.float32))
        return X_train, y_train, X_valid, y_valid, X_test, y_test

    def gas(self):
        target = 'Class'
        data = pd.read_csv('./data/gas/gas.csv')
        data[target] -= 1
        if self.sub == True:
            choose_train_idx = pd.read_pickle('./data/gas/choose_train_idx.pkl')[0].values
            train_valid = data.iloc[choose_train_idx, :]
            X_test = data.drop(choose_train_idx)
            X_train = train_valid
            X_valid = X_test.copy()
        else:
            train_idx = pd.read_pickle('./data/gas/train_idx.pkl')[0].values
            X_train = data.iloc[train_idx, :]
            valid_idx = pd.read_pickle('./data/gas/valid_idx.pkl')[0].values
            X_valid = data.iloc[valid_idx, :]
            test_idx = pd.read_pickle('./data/gas/test_idx.pkl')[0].values
            X_test = data.iloc[test_idx, :]

        y_train, y_valid, y_test = X_train[target].values, X_valid[target].values, X_test[target].values
        X_train = X_train.drop([target], axis=1).values
        X_valid = X_valid.drop([target], axis=1).values
        X_test = X_test.drop([target], axis=1).values

        X_train, X_valid, X_test = quantile_transform(X_train, X_valid, X_test)

        return X_train, y_train, X_valid, y_valid, X_test, y_test

    def diabetes(self):
        target = 'Outcome'
        data = pd.read_csv('./data/diabetes/diabetes-dataset.csv')
        if self.sub == True:
            choose_train_idx = pd.read_pickle('./data/diabetes/choose_train_idx.pkl')[0].values
            train_valid = data.iloc[choose_train_idx, :]
            X_test = data.drop(choose_train_idx)
            X_train = train_valid
            X_valid = X_test.copy()
        else:
            train_idx = pd.read_pickle('./data/diabetes/train_idx.pkl')[0].values
            X_train = data.iloc[train_idx, :]
            valid_idx = pd.read_pickle('./data/diabetes/valid_idx.pkl')[0].values
            X_valid = data.iloc[valid_idx, :]
            test_idx = pd.read_pickle('./data/diabetes/test_idx.pkl')[0].values
            X_test = data.iloc[test_idx, :]

        y_train, y_valid, y_test = X_train[target].values, X_valid[target].values, X_test[target].values
        X_train = X_train.drop([target], axis=1).values
        X_valid = X_valid.drop([target], axis=1).values
        X_test = X_test.drop([target], axis=1).values

        X_train, X_valid, X_test = quantile_transform(X_train, X_valid, X_test)

        return X_train, y_train, X_valid, y_valid, X_test, y_test

    def eeg_eye(self):
        target = 'Class'
        data = pd.read_csv('./data/eeg_eye/eeg-eye.csv')
        data[target] -= 1
        if self.sub == True:
            choose_train_idx = pd.read_pickle('./data/diabetes/choose_train_idx.pkl')[0].values
            train_valid = data.iloc[choose_train_idx, :]
            X_test = data.drop(choose_train_idx)
            X_train = train_valid
            X_valid = X_test.copy()
        else:
            train_idx = pd.read_pickle('./data/eeg_eye/train_idx.pkl')[0].values
            X_train = data.iloc[train_idx, :]
            valid_idx = pd.read_pickle('./data/eeg_eye/valid_idx.pkl')[0].values
            X_valid = data.iloc[valid_idx, :]
            test_idx = pd.read_pickle('./data/eeg_eye/test_idx.pkl')[0].values
            X_test = data.iloc[test_idx, :]

        y_train, y_valid, y_test = X_train[target].values, X_valid[target].values, X_test[target].values
        X_train = X_train.drop([target], axis=1).values
        X_valid = X_valid.drop([target], axis=1).values
        X_test = X_test.drop([target], axis=1).values
        X_train, X_valid, X_test = quantile_transform(X_train, X_valid, X_test)

        return X_train, y_train, X_valid, y_valid, X_test, y_test

    def heart(self):
        data = pd.read_csv('./data/heart/heart_failure_clinical_records_dataset.csv')
        target = 'DEATH_EVENT'
        if self.sub == True:
            choose_train_idx = pd.read_pickle('./data/heart/choose_train_idx.pkl')[0].values
            train_valid = data.iloc[choose_train_idx, :]
            X_test = data.drop(choose_train_idx)
            X_train = train_valid
            X_valid = X_test.copy()
        else:
            train_idx = pd.read_pickle('./data/heart/train_idx.pkl')[0].values
            X_train = data.iloc[train_idx, :]
            valid_idx = pd.read_pickle('./data/heart/valid_idx.pkl')[0].values
            X_valid = data.iloc[valid_idx, :]
            test_idx = pd.read_pickle('./data/heart/test_idx.pkl')[0].values
            X_test = data.iloc[test_idx, :]

        y_train, y_valid, y_test = X_train[target].values, X_valid[target].values, X_test[target].values
        X_train = X_train.drop([target], axis=1).values
        X_valid = X_valid.drop([target], axis=1).values
        X_test = X_test.drop([target], axis=1).values

        X_train, X_valid, X_test = quantile_transform(X_train, X_valid, X_test)
        return X_train, y_train, X_valid, y_valid, X_test, y_test

    def hill(self):
        target = 'Class'
        data = pd.read_csv('./data/hill/hill_vally.csv')
        if self.sub == True:
            choose_train_idx = pd.read_pickle('./data/hill/choose_train_idx.pkl')[0].values
            train_valid = data.iloc[choose_train_idx, :]
            X_test = data.drop(choose_train_idx)
            X_train = train_valid
            X_valid = X_test.copy()
        else:
            train_idx = pd.read_pickle('./data/hill/train_idx.pkl')[0].values
            X_train = data.iloc[train_idx, :]
            valid_idx = pd.read_pickle('./data/hill/valid_idx.pkl')[0].values
            X_valid = data.iloc[valid_idx, :]
            test_idx = pd.read_pickle('./data/hill/test_idx.pkl')[0].values
            X_test = data.iloc[test_idx, :]


        y_train, y_valid, y_test = X_train[target].values, X_valid[target].values, X_test[target].values

        X_train = X_train.drop([target], axis=1).values
        X_valid = X_valid.drop([target], axis=1).values
        X_test = X_test.drop([target], axis=1).values

        X_train, X_valid, X_test = quantile_transform(X_train, X_valid, X_test)
        return X_train, y_train, X_valid, y_valid, X_test, y_test

    def higgs_small(self):
        X_train = np.load('./data/higgs_small/N_train.npy')
        X_valid = np.load('./data/higgs_small/N_val.npy')
        X_test = np.load('./data/higgs_small/N_test.npy')
        y_train = np.load('./data/higgs_small/y_train.npy')
        y_valid = np.load('./data/higgs_small/y_val.npy')
        y_test = np.load('./data/higgs_small/y_test.npy')
        non_nan_index = ~np.isnan(X_train).any(axis=1)
        X_train = X_train[non_nan_index, :]
        y_train = y_train[non_nan_index]
        if self.sub == True:
            data = pd.DataFrame(np.concatenate([X_train, X_valid, X_test], axis=0))
            label = pd.DataFrame(np.concatenate([y_train, y_valid, y_test], axis=0))
            choose_train_idx = pd.read_pickle('./data/higgs_small/choose_train_idx.pkl')[0].values
            train_valid = data.iloc[choose_train_idx, :].values
            y_train = label.iloc[choose_train_idx].values.squeeze()
            X_test = data.drop(choose_train_idx).values
            y_test = label.drop(choose_train_idx).values.squeeze()
            X_train = train_valid
            X_valid, y_valid = X_test.copy(), y_test.copy()
        X_train, X_valid, X_test = quantile_transform(X_train, X_valid, X_test)
        return X_train, y_train, X_valid, y_valid, X_test, y_test


def get_data(datasetname, sub=False):
    if sub == True:
        print('Use T-SNE subset')
    data_pool = Dataset(sub)
    return getattr(data_pool, datasetname)()

def create_missing_value(data, threshold=0.2):
    missing_data = np.copy(data)
    mean = np.mean(data, axis=0)
    missing_space = int(missing_data.size * threshold)
    # for i in range(data.shape[0]):
    #     ratio = random.randint(1, 2)
    #     choose = random.choices(range(data.shape[1]),k=ratio)
    #     missing_data[i, choose] = np.nan
    for _ in range(missing_space):
        rand_x = random.randint(0, missing_data.shape[0]-1)
        rand_y = random.randint(0, missing_data.shape[1]-1)
        missing_data[rand_x, rand_y] = np.nan
    indicator = MissingIndicator(missing_values=np.nan)
    ind = indicator.fit_transform(missing_data)
    missing_data = pd.DataFrame(missing_data)
    for i, m in enumerate(mean):
        col = missing_data.loc[:, i]
        col = col.fillna(m)
        missing_data.loc[:, i] = col
    # missing_data.to_csv('./data/diabetes/missing_data.csv')
    # missing_data = pd.read_csv('./data/missing_data.csv').values
    # ind = pd.read_csv('./data/missing_ind.csv').values
    # ind = (1 - ind).astype(bool)
    return missing_data.values, ind.astype(np.int)

if __name__ == '__main__':
    a = get_data('epsilon')
    print('a')