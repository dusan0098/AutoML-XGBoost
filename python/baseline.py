""""
File contents:
Code provided in the original code skeleton for the project
Changes to original code:
    - Added n_thread parameter to init_model function of the XGBoostTest class, this was done to try and speed up the evaluations on test datasets
    - Added evaluate_baseline() - function for getting the baseline performance (XGBoost running on default config) for all test tasks
    - Added get_local_changes() - function for getting performance when changing each parameter by different increments
"""
import os
import time
import numpy as np
import xgboost as xgb
from IPython.core.display_functions import display
from sklearn.metrics import roc_auc_score
import pandas as pd

from project_utils import get_task_metafeatures, load_test_data, meta_feature_names, \
    default_config, test_ids, get_best_config_per_task, get_all_metafeatures, \
    hyperparameters_data, make_valid_config


class XGBoostTest:
    """Set seed, get train-test data and meta features of task_id"""

    def __init__(self, task_id: int, meta_feature_names: list, seed: int = 12345):
        self.seed = seed
        self.task_id = task_id
        self.train_x, self.train_y, self.test_x, self.test_y, self.nclasses = load_test_data(task_id)
        self.meta_features = get_task_metafeatures(task_id, meta_feature_names)

    """Initialize the xgboost learner based on a hyperparameter configuration"""

    def init_model(self, config: dict, seed: int = None):
        xgb.set_config(verbosity=0)
        rng = np.random.RandomState(self.seed) if seed is None else np.random.RandomState(seed)
        # added threads for speedup
        extra_args = dict(random_state=rng, eval_metric=roc_auc_score, nthread=4)
        if self.nclasses > 2:
            extra_args["objective"] = "multi:softmax"
            extra_args.update({"num_class": self.nclasses})

        model = xgb.XGBClassifier(
            **config,
            **extra_args
        )
        return model

    """
    Evaluate the xbgoost learner by first initializing the model based on a hyperparameter configuration,
    train on train set and evaluate on test set based on AUC
    """

    def evaluate(self, config: dict, seed: int = None):
        model = self.init_model(config, seed)
        train_start = time.time()
        model.fit(self.train_x, self.train_y)
        timetrain = time.time() - train_start
        prediction_prob = model.predict_proba(self.test_x)
        if self.nclasses == 2:
            # handling binary classification
            prediction_prob = prediction_prob[:, 1]
        auc = roc_auc_score(self.test_y, prediction_prob, multi_class="ovr")
        return auc, timetrain


"""
   Evaluating baseline performance for test tasks - if all have already been evaluated for loop will be skipped
   results in: ./data/baseline_performance.csv
"""


def evaluate_baseline():
    seed = 0

    if os.path.isfile('./data/baseline_performance.csv'):
        baseline_performance = pd.read_csv('./data/baseline_performance.csv')
    else:
        baseline_performance = pd.DataFrame()

    evaluated_ids = list(baseline_performance['task_id'].astype(int))

    # Checking if any test tasks are missing in the CSV
    nonevaluated_ids = list(set(test_ids) - set(evaluated_ids))

    """Evaluating performance of default config for each test task"""
    for task_id in nonevaluated_ids:
        print(f"Task ID: {task_id}")
        # initialize the interface to the XGBoost test tasks
        objective = XGBoostTest(task_id, meta_feature_names=meta_feature_names, seed=seed)
        meta_features = objective.meta_features
        print(f"The meta features of task ID {task_id} are:")
        print(meta_features)
        # modelling and evaluating the default config
        auc, timetrain = objective.evaluate(default_config)
        print(f"The default config scored an AUC of {auc} in {timetrain}s on task ID {task_id}.")

        current_task_performance = {'task_id': int(task_id), 'auc': auc, 'timetrain': timetrain}
        baseline_performance = baseline_performance.append(current_task_performance, ignore_index=True)

        # updated after every training to save time in case of failure/long run time
        baseline_performance.to_csv('./data/baseline_performance.csv', index=False)

    display(baseline_performance)


"""
    Evaluating performance when changing each hyperparameter by 1,5 or 10% individually on all test tasks
    results in: ./data/local_changes.csv
    10 tasks * 10 parameters * 6 delta values = 600 evaluations
    These are NOT used for training any model, just to see if there is a clear better default config
"""


def get_local_changes():
    deltas = [1, -1, 5, -5, 10, -10]
    parameters = hyperparameters_data
    seed = 0

    # data:num_rounds => XGBoost API n_estimators
    parameters[0] = 'n_estimators'
    baseline = pd.read_csv('./data/baseline_performance.csv')

    """
    Evaluating all possible combinations for all datasets would take too long. We look at the changes on those tasks that finished training in under 10 seconds with the default config
    """
    fast_ids = baseline[baseline['timetrain'] <= 10]['task_id'].to_list()

    if os.path.isfile('./data/local_changes.csv'):
        local_changes = pd.read_csv('./data/local_changes.csv')
    else:
        local_changes = pd.DataFrame()

    evaluated_ids = list(local_changes['task'].astype(int))

    # Checking if any tasks are missing in the CSV
    nonevaluated_ids = list(set(fast_ids) - set(evaluated_ids))
    print("Tasks that need to be evaulated:", nonevaluated_ids)

    # For each (task,parameter) evaluate new AUC when the parameter is increased/decreased by 1,5,10%
    for task_id in nonevaluated_ids:
        base_performance = baseline[baseline['task_id'] == task_id].head(1)
        base_auc = base_performance['auc'].item()
        base_time = base_performance['timetrain'].item()
        print("Evaluating all changes for task: ", task_id)
        for param in parameters:
            print("Evaluating changes on:", param)
            for delta in deltas:
                # new_row contains the information of the AUC and time achived with the new configuration
                new_row = {"task": task_id, "parameter": param, "delta(%)": delta}
                config = default_config.copy()
                config[param] = config[param] * (float(100 + delta)) / 100
                config = make_valid_config(config)
                new_row['new_value'] = config[param]

                # Evaluating model with new parameter value
                objective = XGBoostTest(task_id, meta_feature_names=meta_feature_names, seed=seed)
                auc, timetrain = objective.evaluate(config)
                new_row.update({'new_auc': auc, 'new_time': timetrain})
                new_row['better_auc'] = auc >= base_auc
                new_row['better_time'] = timetrain <= base_time
                # Adding the row to the local_changes table
                new_row = pd.DataFrame([new_row])
                local_changes = pd.concat([local_changes, new_row], ignore_index=True)
        local_changes.to_csv('./data/local_changes.csv', index=False)

    display(local_changes)


if __name__ == "__main__":
    """Evaluating performance for default configuration"""
    evaluate_baseline()
    get_local_changes()
