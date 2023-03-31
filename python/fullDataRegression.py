"""
File contents:
    Data preparation, model training and evaluation methods
    Model: F(metafeature, hyperparameters) = AUC
    When a new dataset arrives we fix the regressors metafeature input and try out random samples for hyperparameters
    The combination that achieves the highest predicted AUC is then evaulated on the test dataset
"""
import os

from sklearn.gaussian_process import GaussianProcessRegressor
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from sklearn.model_selection import RepeatedKFold
import pickle
from sklearn.metrics import r2_score
import pandas
import pandas as pd
import numpy as np

from python import project_utils
from python.project_utils import get_dataset_to_task, training_meta_features, hyperparameters_data, test_ids, \
    make_valid_config, meta_feature_names, default_config
from datetime import datetime
from skopt.space import Real, Integer
from python.taskRegression import sample_points
from python.baseline import XGBoostTest
import scipy

"""
    Used to have some sort of feedback during long Bayesian Optimisation runs
"""


def bayes_callback(res):
    print("Next iteration, Time:", datetime.now().strftime("%H:%M:%S"))


"""
    Preparing the data for the regressor F(metafeatures,hyperparameters) = AUC
"""


def prepare_full_data(address='./data/joint_and_averaged_clean.csv'):
    avg_perf = pd.read_csv('./data/average_performance.csv')
    meta_train = pd.read_csv('./data/features.csv')
    meta_train['MaxNominalAttDistinctValues'].hist(bins=100)
    print(meta_train.shape, avg_perf.shape)

    joint_and_averaged = avg_perf.merge(meta_train, how='left', on='data_id')
    joint_and_averaged['MaxNominalAttDistinctValues'] = joint_and_averaged['MaxNominalAttDistinctValues'].fillna(5)
    joint_and_averaged.drop(['index', 'name', 'version', 'status'], axis=1, inplace=True)
    joint_and_averaged['task_id'] = joint_and_averaged['data_id'].map(get_dataset_to_task())
    joint_and_averaged.to_csv(address)


"""
    Training the regressor to predict AUC
"""


def compare_full_data_regressor(data_address='./data/joint_and_averaged_clean.csv', fraction=0.15, n_iter=20):
    full_data = pandas.read_csv(data_address)
    """Data will be structured as
    X_train: dataset_metafeatures, XGBoost hyperparameters
    y_train: avg_auc
    We sample 15% of the data to make the training time ~2 hours
    """
    sampled_data = full_data.sample(frac=fraction)

    # Defining search space for XGBoost regressor
    search_space = {}
    search_space['eta'] = Real(0, 1, 'uniform')
    search_space['n_estimators'] = Integer(50, 500)
    search_space['max_depth'] = Integer(3, 15)
    search_space['learning_rate'] = Real(10e-4, 0.1, 'log-uniform')
    search_space['colsample_bytree'] = Real(0.3, 0.9, 'uniform')
    search_space['subsample'] = Real(0.3, 1, 'uniform')

    # Selecting relevant features for regressor
    training_features = training_meta_features + hyperparameters_data
    X = sampled_data[training_features].to_numpy()
    y = sampled_data['avg_auc'].to_numpy()

    # We split the data 2:1 to get 10% of the entire dataset in each training
    cv_outer = RepeatedKFold(n_splits=3, n_repeats=1, random_state=1)
    # Lists for nested sampling results
    outer_score = []
    regressors = []
    search_results = []

    """Outer loop 3-fold cross validation, where each fold is trained on 10% of the original dataset (we sampled 15%)"""
    for counter, (train_index, test_index) in enumerate(cv_outer.split(X, y)):
        print("Starting outer evaluation ", counter + 1, "for Bayesian optimisation \n Time:",
              datetime.now().strftime("%H:%M:%S"))

        # selecting relevant data for fold
        X_train, X_test = X[train_index, :], X[test_index, :]
        y_train, y_test = y[train_index], y[test_index]

        """
        Inner loop uses 5-fold cross validation to evaluate regressor configurations
        Metric doesn't need to be specified - inherited by XGBoost (standard R2 score)
        """
        cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=1)
        search = BayesSearchCV(estimator=XGBRegressor(), search_spaces=search_space, n_jobs=-1,
                               cv=cv, n_iter=n_iter)  # n_iter=50 default
        # perform the search and save results
        curr_result = search.fit(X_train, y_train, callback=bayes_callback)

        search_results.append(curr_result)
        best_candidate = curr_result.best_estimator_
        regressors.append(best_candidate)
        # report the best result
        print("Best score in this iteration:", search.best_score_)
        print("Best regression parameters in this iteration:", search.best_params_)

        y_pred = best_candidate.predict(X_test)
        score = r2_score(y_test, y_pred)
        outer_score.append(score)

    # Save results
    with open('full_data_xgboost_result.pkl','wb') as f:
        pickle.dump(outer_score, f)
    with open('full_data_xgboost_regressor.pkl', 'wb') as f:
        pickle.dump(regressors, f)
    with open('full_data_xgboost_search.pkl', 'wb') as f:
        pickle.dump(search_results, f)


    return search_results,regressors,outer_score


def evaluate_full_data_regression(save_address="./data/fulldata_nested_xgboost_results.csv", n_points=1000,
                                  use_gaussian=False, fraction = 0.15, n_iter = 20):
    """
    Loading previous BO results for evauluation on test tasks
    """
    if os.path.isfile('full_data_xgboost_regressor.pkl'):
        print("Loading models from pkl - This will take at least a minute\n")
        with open('full_data_xgboost_result.pkl', 'rb') as f:
            scores = pickle.load(f)
        with open('full_data_xgboost_regressor.pkl', 'rb') as f:
            regressors = pickle.load(f)
        with open('full_data_xgboost_search.pkl', 'rb') as f:
           searches = pickle.load(f)
    else:
        searches, regressors, scores = compare_full_data_regressor(fraction=fraction ,n_iter=n_iter)

    index_max = np.argmax(scores)
    best_regressor = regressors[index_max]
    best_search = searches[index_max]
    results_dict = best_search.cv_results_

    test_features = pandas.read_csv('./data/test_features.csv')

    """
    For each of the test tasks we will predict the best configuration by sampling random configurations and observing the expected AUC
    """
    print("Generating random configurations")
    sample_configs = sample_points(n_points=n_points)[project_utils.hyperparameters_data].to_numpy()

    predicted_config = {}
    print("Selecting the best configuration for each task:")
    for test_id in test_ids:
        print("Task:", test_id, "time", datetime.now().strftime("%H:%M:%S"))
        # Constructing input for regressor based on current dataset
        meta_featues = test_features[test_features['task_id'] == test_id].head(1)[training_meta_features].to_numpy()
        helper_materix = np.tile(meta_featues, (sample_configs.shape[0], 1))
        input = np.concatenate((helper_materix, sample_configs), axis=1)
        predicted_auc = best_regressor.predict(input)
        #print(predicted_auc)

        # Getting the best hyperparameter config for current dataset
        if use_gaussian:
            """
                Instead of just picking the random point with the best AUC we can further model the AUC for any point by 
                using a second round of regression - Here with a Gaussian Process
            """
            gpr_estimator = GaussianProcessRegressor().fit(sample_configs, predicted_auc)
            # We use the default config as a starting point
            print("Finding best configuration for task: ", test_id, " using Gaussian process")
            default_point = [464, 0.0082, 0, 0.06068, 0.00235, 0.982, 11, 3.30, 0.975, 0.9]
            bounds = ((0,1500.0),(0,1.0),(0,1.0),(0,1.0),(0,1.0),(0,1),(0,20.0),(0,10.0),(0,1.0),(0,1.0))
            optimisation_results = scipy.optimize.minimize(lambda x: (-1) * gpr_estimator.predict(np.array([x])),x0=default_point, method='SLSQP',
                                                           bounds = bounds)
            predicted_config[test_id] = optimisation_results.x
            #print(predicted_config)
        else:
            ind = np.argmax(predicted_auc)
            predicted_config[test_id] = sample_configs[ind, :]

    baseline = pd.read_csv('./data/baseline_performance.csv')
    comparison = pd.DataFrame()
    for key, config in predicted_config.items():
        """For each of the Test tasks (18 in total) we evaluate the performance of its predicted best config for XGBoost"""
        current_task = key
        print("Evaluating task with id: ", current_task)
        seed = 123
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

        """Corrections to continuous predictions that might not be possible"""
        new_config = dict(zip(hyperparameters_data, config))
        new_config = make_valid_config(new_config)

        """Evaluating new config performance"""
        objective = XGBoostTest(current_task, meta_feature_names=meta_feature_names, seed=seed)
        auc, traintime = objective.evaluate(new_config)

        """Adding info about new performance """
        new_row['new_auc'] = auc
        print("New auc", auc)
        new_row['new_time'] = traintime
        new_row['delta_auc'] = new_row['new_auc'] - new_row['base_auc']
        new_row['better_auc'] = new_row['new_auc'] > new_row['base_auc']
        new_row['better_time'] = new_row['new_time'] < new_row['base_time']
        new_row = pd.DataFrame([new_row])
        comparison = pd.concat([comparison, new_row], ignore_index=True)
    #comparison.to_csv(save_address)
    return comparison

if __name__ == "__main__":
    # Uncomment if you want to build the CSV again
    # prepare_full_data()

    """
        Uncomment if you want to run the Bayesian optimisation from scratch
        Note - if the training is too slow lower the fraction (for data sampling) and n_iter (for BO)    
    """
    # compare_full_data_regressor(fraction=0.15 ,n_iter=20)

    """
        Samples random configrations for XGBoost and predicts the expected AUC using the trained model, 
        evaluates the best prediction for each test task.
    """

    evaluate_full_data_regression(save_address="./data/fulldata_nested_xgboost_results.csv", n_points=200000, use_gaussian = False, fraction=0.15, n_iter=20)

    # evaluate_full_data_regression(save_address="./data/joint_with_GP.csv", n_points=5000,
    #                                use_gaussian=True)
