from DAN_Task import DANetClassifier
import os
import numpy as np
import torch
import argparse
from qhoptim.pyt import QHAdam
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from numpy.random import RandomState
from data.dataset import get_data
from hyperopt.hp import uniform, randint, choice
from hyperopt import Trials, STATUS_OK, partial, tpe, fmin
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score

def save_params(params):
    if os.path.exists(os.path.join('./logs', args.dataset, args.experiment)):
        mode = 'a'
    else:
        os.makedirs(os.path.join('./logs', args.dataset, args.experiment))
        mode = 'w'
    file = open(os.path.join('./logs', args.dataset, args.experiment, 'hyper_params.txt'), mode)
    for k, v in params.items():
        file.write(str(k) + ': ' + str(v) + ', ')

    file.write('\n')
    file.close()

def data(dataset_name):
    X_train, y_train, X_valid, y_valid, X_test, y_test = get_data(dataset_name, sub=False)
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def hyperopt_objective(params):
    print(params)
    save_params(params)
    torch.cuda.empty_cache()
    clf = DANetClassifier(
        optimizer_fn=QHAdam,
        optimizer_params=dict(lr=params['lr'], weight_decay=1e-5, nus=(0.7, 0.99), betas=(0.95, 0.998)),
        scheduler_params=dict(gamma=0.95, step_size=20),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        sub_class=params['sub_class'],
        init_dim=params['init_dim'],
        primary_capsule_size=params['primary_capsule_size'],
        digit_capsule_size=params['digit_capsule_size'],
        leaves=params['leaves'],
        decode=decode,
        seed=1,
    )
    if metric == 'acc':
        eval_metric = ['accuracy']
    elif metric == 'auc':
        eval_metric = ['auc']
    else:
        eval_metric = ['logloss']

    clf.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_test, y_test), (X_valid, y_valid)],
        eval_name=['test', 'valid'],
        eval_metric=eval_metric,
        max_epochs=7000, patience=100,
        batch_size=2048, virtual_batch_size=256,
        logname=os.path.join(args.dataset, args.experiment, str(params)),
        device_id=gpu_id
    )
    best_value = clf.best_cost
    y_true, y_pred, reconstruct_data = clf.predict(X_test, y_test)
    if metric == 'acc':
        test_value = accuracy_score(y_true, np.argmax(y_pred, axis=1))
    elif metric == 'auc':
        test_value = roc_auc_score(y_true, y_pred[:, 1])
    else:
        test_value = log_loss(y_true, y_pred)

    print(params)
    print('Best ' + eval_metric[-1] + ': {:.5f}'.format(best_value))
    print('Test value: {:.5f}'.format(test_value))

    mode = 'a' if os.path.exists(os.path.join('./logs', args.dataset, args.experiment, 'hyper_params.txt')) else 'w'
    file = open(os.path.join('./logs', args.dataset, args.experiment, 'hyper_params.txt'), mode)
    file.write('Best value: ' + str(best_value) + '\n')
    file.write('Test value: ' + str(test_value) + '\n\n')
    file.close()
    if clf._callback_container.callbacks[1].is_maximize == True:
        return -best_value
    else:
        return best_value

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch v1.12, TabCaps classfication Training')
    parser.add_argument('-d', '--dataset', type=str, required=False, default='gas', help='Dataset Name')
    parser.add_argument('-r', '--decode', action='store_true', default=True, help='Reconstruct dataset')
    parser.add_argument('-t', '--tsne', action='store_true', default=False, help='Train tsne subset')
    parser.add_argument('-e', '--experiment', type=str, required=False, default='', help='experiment Name')
    parser.add_argument('-v', '--metric', type=str, required=False, default='acc', help='using metric')
    parser.add_argument('-g', '--gpu_id', type=str, default='2', help='GPU ID')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    gpu_id = args.gpu_id
    dataset_name = args.dataset
    decode = args.decode
    sub = args.tsne
    metric = args.metric
    print('===> Getting data ...')
    print('Dataset: ' + dataset_name)
    X_train, y_train, X_valid, y_valid, X_test, y_test = get_data(dataset_name, sub=sub)
    params_space = {
                    'lr': uniform('lr', 4e-2, 2e-1),
                    'sub_class': randint('sub_class', 1, 5),
                    'init_dim': randint('init_dim', 32, 128),
                    'primary_capsule_size': randint('primary_capsule_size', 4, 32),
                    'digit_capsule_size': randint('digit_capsule_size', 4, 32),
                    'leaves': randint('leaves', 16, 64)
                    }
    trials = Trials()
    best = fmin(
        hyperopt_objective,
        space=params_space,
        algo=partial(tpe.suggest, n_startup_jobs=1),
        trials=trials,
        max_evals=50,
        rstate=RandomState(seed=14)
    )
    print("Best performing model chosen hyper-parameters:")
    print(best)