"""
File contents:
    Functions for training and evaluating taskwise regression models.
    Method: For each of the 94 training tasks we fit regression models using the ENTIRE average performance data (about 650.000 rows)
            get_taskwise_regressors() - For each of the 94 datasets we try to learn a mapping F(hyperparameters) = AUC
            After these models are trained we sample random points in the XGBoost search space and evaluate the expected AUC for each of the models
            The final predicted AUC for a new task is the weighted average of those predictions based on the L2 distance of the new task to the training tasks
"""

from collections import defaultdict
from datetime import datetime
import pickle
from random import uniform

import numpy as np
import pandas as pd
from IPython.core.display_functions import display
from numpy.random import random

from python.baseline import XGBoostTest
from python.project_utils import training_meta_features, hyperparameters_data, get_average_performance, \
    get_dataset_to_task, get_best_config_per_task, make_valid_config, meta_feature_names, test_ids
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from sklearn.model_selection import RepeatedKFold
from skopt.space import Real, Integer
import os.path
from scipy.spatial.distance import pdist, squareform

"""
Crates a XGBoost Regressor for predicting the AUC for each of the 94 training tasks
Returns a dictionary where dict[6] is the AUC regression model trained on data from task with task_id = 6
NOTE - WITH CURRENT PARAMETERS TRAINING ALL MODELS TOOK AROUND 4 HOURS
 """
def get_taskwise_regressors(address="./data/dummy_file.pkl"):
    avg_perf = get_average_performance(per_task=True)

    if os.path.isfile(address):
        print("File already exists, loading from pickle :\n")
        with open(address, 'rb') as f:
            models = pickle.load(f)
            print("Regressors loaded\n")
            return models

    # Defining search space for model
    search_space = {}
    search_space['eta'] = Real(0, 1, 'uniform')
    search_space['n_estimators'] = Integer(50, 500)
    search_space['max_depth'] = Integer(3, 15)
    search_space['learning_rate'] = Real(10e-4, 0.1, 'log-uniform')
    search_space['colsample_bytree'] = Real(0.3, 0.9, 'uniform')
    search_space['subsample'] = Real(0.3, 1, 'uniform')

    # Defining CV method and Bayesian optimisation proceduree
    CV_folds = RepeatedKFold(n_splits=4, n_repeats=1, random_state=1)
    search = BayesSearchCV(estimator=XGBRegressor(), search_spaces=search_space, n_jobs=-1, cv=CV_folds, n_iter=5)

    models = {}
    for key in avg_perf.keys():
        print("Evaluating regressor for dataset with task_id: ", key)
        print("Start time: ", datetime.now().strftime("%H:%M:%S"))
        #Retrieving average performance data for current training task
        current_df = avg_perf[key]

        # Selecting relevant information for regressor
        features_train = current_df[hyperparameters_data]
        target_train = current_df['avg_auc']

        # Performing the search
        search.fit(features_train.to_numpy(), target_train.to_numpy())
        # report the best result
        print(search.best_score_)
        print(search.best_params_)
        # train the model on the best parameters
        model = XGBRegressor(**search.best_params_)

        # add the trained model to our collection
        model.fit(features_train.to_numpy(), target_train.to_numpy())
        models[key] = model

    # save all models (done in every loop to not lose progress)
    with open('taskwise_models.pkl', 'wb') as f:
        pickle.dump(models, f)
    return models


"""
Cacluated Euclidian distance between the training and test tasks - used for weighing the taskwise predictions of the AUC
"""
def get_train_test_distances():
    train_points, test_points = get_cleaned_metafeatures()
    """
    Calculting distances between datasets based on their metafeatures
    """
    df_distances = pd.DataFrame()
    for _, test_row in test_points.iterrows():
        test_id = test_row['task_id']
        test_features = test_row[training_meta_features]

        for _, train_row in train_points.iterrows():
            train_id = train_row['task_id']
            train_features = train_row[training_meta_features]
            dist = np.linalg.norm(train_features.values - test_features.values)
            df_distances = df_distances.append({'test_id': test_id, 'train_id': train_id, 'dist': dist}, ignore_index=True)

    df_distances.sort_values(['test_id', 'train_id'], ascending=[True, True])

    distance_dict = defaultdict(dict)
    for i, row in df_distances.iterrows():
        distance_dict[row.test_id][row.train_id] = row.drop(['test_id', 'train_id']).to_dict()
    return distance_dict

"""
    Function for preparing metafeatures for the taskwise regressors
"""
def get_cleaned_metafeatures():
    meta_features = training_meta_features.copy()
    features = ['task_id'] + meta_features

    train_points = pd.read_csv('./data/features.csv')
    test_points = pd.read_csv('./data/test_features.csv')
    train_points['task_id'] = train_points['data_id'].map(get_dataset_to_task())
    test_points['task_id'] = test_points['task_id'].astype(int)

    # Value used during imputation in previous aproaches
    train_points['MaxNominalAttDistinctValues'] = train_points['MaxNominalAttDistinctValues'].fillna(5)
    test_points['MaxNominalAttDistinctValues'] = test_points['MaxNominalAttDistinctValues'].fillna(5)
    train_points = train_points[features]
    test_points = test_points[features]

    return train_points, test_points

"""
    Samples random configurations for XGBoost, limits are based on the intervals seen in good HPCs from xgboost_meta_data.csv
"""
def sample_points(n_points = 500):
    config_ids = list(range(n_points))
    #simulating log-uniform distribution num_rounds (20,5000)
    num_rounds = np.exp(np.random.uniform(low=3, high=8, size=n_points)).astype(int)

    etas = np.random.uniform(low=10e-4, high=1.0, size=n_points)
    gammas = np.random.uniform(low=10e-5, high=5.0, size=n_points)
    lambdas = np.random.uniform(low=10e-5, high=400, size=n_points)
    alphas = np.random.uniform(low=10e-6, high=20, size=n_points)
    subsamples = np.random.uniform(low=0.15, high=1, size=n_points)
    max_depths = np.random.uniform(low=2, high=20, size=n_points).astype(int)
    min_child_weights = np.random.uniform(low=0.03, high=55, size=n_points)
    colsample_bytrees = np.random.uniform(low=10e-2, high=1.0, size=n_points)
    colsample_bylevels = np.random.uniform(low=10e-2, high=1.0, size=n_points)

    samples = pd.DataFrame(data=[config_ids, num_rounds, etas, gammas, lambdas, alphas, subsamples,max_depths, min_child_weights,colsample_bytrees, colsample_bylevels ]).T
    samples.columns = ['config_id'] + hyperparameters_data
    return samples


"""
    Evaluates expected AUC of a HPC on a new dataset. For each test dataset returns the HPC with the highest expected
    AUC.
"""
def get_best_candidate(n_points = 50):
    points = sample_points(n_points)
    # regressors[task_id] is XGBoost classifier trained on task_id
    regressors = get_taskwise_regressors('taskwise_models.pkl')
    #We use the distances in the metafeatures space as weights when predicting the AUC
    distances = get_train_test_distances()

    best_candidates = {}
    pred = {}
    #Solid speedup we only run the predictions once instead of for all 18 datasets
    print("Calculating expected AUCs for each candidate, time:",datetime.now().strftime("%H:%M:%S"))
    for index, point in points[hyperparameters_data].iterrows():
        pred[index] = {}
        config = point.values.tolist()
        if index % 200 == 0:
            print("Canidates evaluated:", index, " time:",datetime.now().strftime("%H:%M:%S"))
        for train_id, model in regressors.items():
            #Represents a points predicted AUC by the Regressor with id = train_id
            pred[index][train_id] = model.predict([config])

    for test_id in test_ids:
        best_avg = 0
        best_candidate = None
        print("Starting evaluation of configs for task:", test_id, " time:",datetime.now().strftime("%H:%M:%S"))

        for index, point in points[hyperparameters_data].iterrows():
            avg_auc = 0
            if index % 500 == 0:
                print("Candidate:", index)
            for train_id, _ in regressors.items():
                avg_auc += pred[index][train_id]/distances[test_id][train_id]['dist']

            if avg_auc > best_avg:
                best_avg = avg_auc
                best_candidate = point
        print("Evaluation over, time:", datetime.now().strftime("%H:%M:%S"))
        best_candidates[test_id] = best_candidate.to_dict()

    #Saving best candidates dictionary in each iteration
    with open('best_candidates.pkl', 'wb') as f:
        pickle.dump(best_candidates, f)
    return best_candidates

def evaluate_taskwise_regression(n_points = 2000):
    """
        Getting the best candidate for each test dataset
    """
    if os.path.isfile('best_candidates.pkl'):
        with open('best_candidates.pkl', 'rb') as f:
            configs = pickle.load(f)
    else:
        configs = get_best_candidate(n_points=n_points)
    print(configs)

    """
        Evalating the predictions and saving the performance info on test datasets
    """
    baseline = pd.read_csv('./data/baseline_performance.csv')
    comparison = pd.DataFrame()
    for key, config in configs.items():
        """For each of the Test tasks (18 in total) we evaluate the performance of its predicted best config for XGBoost"""
        current_task = key
        new_config = make_valid_config(config)
        print("Evaluating task with id: ", current_task)
        seed = 0
        """We save the performance in a dataframe containing:
        1.task_id
        2.baseline auc and time 
        3.new auc and time - for the predicted best configuration
        """
        new_row = {}
        new_row['task_id'] = current_task
        base_performance = baseline[baseline['task_id'] == current_task].head(1)
        new_row['base_auc'] = base_performance['auc'].values[0]
        new_row['base_time'] = base_performance['timetrain'].values[0]

        """Evaluating new config performance"""
        objective = XGBoostTest(current_task, meta_feature_names=meta_feature_names, seed=seed)
        auc, traintime = objective.evaluate(new_config)

        """Adding info about new performance """
        new_row['new_auc'] = auc
        print("New auc", auc)
        new_row['new_time'] = traintime
        new_row['delta_auc'] = new_row['new_auc'] - new_row['base_auc']
        new_row['better_auc'] = new_row['new_auc'] >= new_row['base_auc']
        new_row['better_time'] = new_row['new_time'] <= new_row['base_time']
        new_row = pd.DataFrame([new_row])
        comparison = pd.concat([comparison, new_row], ignore_index=True)
    # display(comparison)
    # comparison.to_csv('./data/weighted_taskwise_regression_performance.csv')
    return comparison

if __name__ == "__main__":
    taskwise_regressoin_baseline_comparison = evaluate_taskwise_regression(n_points=2000)
