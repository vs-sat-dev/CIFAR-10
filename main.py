import numpy as np
import optuna

import utils
from run import run_model


if __name__ == '__main__':
    meta = utils.unpickle('data/batches.meta')

    data_train = dict()
    data_test = dict()
    for i in range(5):
        batch = utils.unpickle(f'data/data_batch_{i+1}')
        if i == 0:
            data_train['labels'] = np.array(batch[b'labels'], dtype=np.int64)
            data_train['data'] = np.array(batch[b'data'], dtype=np.float32) / 255.0
        else:
            data_train['labels'] = np.concatenate((data_train['labels'], np.array(batch[b'labels'], dtype=np.int64)), axis=0)
            data_train['data'] = np.concatenate((data_train['data'],
                                                 np.array(batch[b'data'], dtype=np.float32) / 255.0), axis=0)
        print(f'shape {data_train["data"].shape}')

    batch = utils.unpickle('data/test_batch')
    data_test['labels'] = np.array(batch[b'labels'], dtype=np.int64)
    data_test['data'] = np.array(batch[b'data'], dtype=np.float32) / 255.0

    optuna_train, optuna_test = dict(), dict()
    optuna_train['data'] = data_train['data'][:20000]
    optuna_train['labels'] = data_train['labels'][:20000]
    optuna_test['data'] = data_train['data'][20000:30000]
    optuna_test['labels'] = data_train['labels'][20000:30000]

    objective = lambda trial: run_model(optuna_train, optuna_test, trial=trial)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    params = study.best_trial.params
    print(f'Params: {params}')

    best_accuracy = run_model(data_train, data_test, params=params)
    print(f'Best accuracy: {best_accuracy}')

