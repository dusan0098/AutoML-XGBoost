"""
File contents:
Functions for training and evaluating different regression models, with and without bayesian optimisation, to predict best values for each of the 10 XGBoost hyperparameters
    - Here all methods try to directly capture the function F(meta_features) = hyperparameters based on the 94 optimal configurations from the training data (configs with best average AUC)
"""
import os
from datetime import datetime
from statistics import mean
import numpy as np
import pandas as pd
from IPython.display import display
from numpy import std
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, RepeatedKFold
from sklearn.svm import SVC, SVR
from skopt import BayesSearchCV
from skopt.space import Categorical, Real, Integer
from xgboost import XGBRegressor

from project_utils import meta_feature_names, test_ids, default_config, get_best_config_per_task
from python.FFN_regression import MyDataset, RMSELoss, SimpleFFN
from sklearn import ensemble
from sklearn.multioutput import MultiOutputRegressor
from python.baseline import XGBoostTest
from python.project_utils import hyperparameters_data, make_valid_config
import torch
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import skopt
import pickle

"""
    1. Trains 10 regressors of the same type (or single regressor if the model is Multioutput by nature) to predict the values of the best configuration using dataset metafeatures
        -Training data - for each of the 94 training tasks we provide: the dataset metafeatures (X_train), the configuration with the best performance on XGBoost based on average AUC (y_train)
        -X_test - represent the metafeatures of the 18 test tasks
    2. Predicts the best configuration for each test task
    3. Evaluates and saves the performance of each configuration
"""

def compare_regressor_with_baseline(regressor=None, custom_model=False):
    """Extracting data needed for training and prediction"""
    my_data = MyDataset()
    X_train = my_data.x_train
    y_train = my_data.y_train
    X_test = my_data.x_test

    """Creating predictions with MultiOutput regression model"""
    # deprecated: model_type in ['random_forest','gradient_boosting','mlperceptron','xgboost']:
    if isinstance(regressor, ensemble.RandomForestRegressor) or isinstance(regressor,ensemble.GradientBoostingRegressor) \
            or isinstance(regressor, MLPRegressor) or isinstance(regressor, XGBRegressor):
        reg = MultiOutputRegressor(regressor)
        reg.fit(X_train, y_train)
        predictions = reg.predict(X_test)

    elif isinstance(regressor, KNeighborsRegressor):
        """These models support multioutput regression by default"""
        reg = regressor
        reg.fit(X_train, y_train)
        predictions = reg.predict(X_test)

    elif custom_model:
        """Scaled RMSE based on each features ranges"""
        loss_function = RMSELoss(my_data.y_range)
        """Feed forward net implemented in FFN_regression.py"""
        model = SimpleFFN(in_features=X_train.size(dim=1), out_features=y_train.size(dim=1), hidden_size=16,
                          num_layers=4, activation='tanh', dropout=0.1)
        model.fit(X_train, y_train, loss_function, 100)
        predictions = model(X_test).detach().numpy()

    baseline = pd.read_csv('./data/baseline_performance.csv')
    # Dataframe containing new performance info
    comparison = pd.DataFrame()
    for j in range(len(predictions)):
        """For each of the Test tasks (18 in total) we evaluate the performance of its predicted best config for XGBoost"""
        current_task = test_ids[j]
        print("Evaluating task with id: ", current_task)
        pred = predictions[j]
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
        """We convert the model predictions into a dictionary for the HPC"""
        new_config = {}
        for i in range(len(hyperparameters_data)):
            new_config[hyperparameters_data[i]] = pred[i]
        """Corrections to continuous predictions that might not be possible"""
        new_config = make_valid_config(new_config)

        """Evaluating new config performance"""
        objective = XGBoostTest(current_task, meta_feature_names=meta_feature_names, seed=seed)
        auc, traintime = objective.evaluate(new_config)

        """Adding info about new performance to table"""
        new_row['new_auc'] = auc
        print("New auc", auc)
        new_row['new_time'] = traintime
        new_row['delta_auc'] = new_row['new_auc'] - new_row['base_auc']
        new_row['better_auc'] = new_row['new_auc'] > new_row['base_auc']
        new_row['better_time'] = new_row['new_time'] < new_row['base_time']
        new_row = pd.DataFrame([new_row])
        comparison = pd.concat([comparison, new_row], ignore_index=True)

    return comparison


"""
    Function to evaluate KNN for different values of K, uses compare_regressor_with_baseline
    Added to reduce clutter in __main__
"""


def evaluate_knn_regressors(values_of_k=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]):
    knn_results = pd.DataFrame()
    for num_neighbours in values_of_k:
        curr_result = compare_regressor_with_baseline(KNeighborsRegressor(n_neighbors=num_neighbours))
        print("For k=", num_neighbours, "the results were: \n")
        display(curr_result)
        curr_result['n_neighbors_knn'] = num_neighbours
        knn_results = knn_results.append(curr_result)
    print("Final results for all K \n")
    display(knn_results)
    # knn_results.to_csv("./data/kmeans_num_neighbours.csv")


"""
    1. Trains 10 regressors of the same type, each with a DIFFERENT configuration (unlike in compare_regressor_with_baseline) to predict the values of the XGBoost hyperparameters - Uses Bayesian Optimisation
    2. Evaluates the predicted best configurations on the 18 test tasks 
    3. Saves the performance information
"""


def evaluate_bayesian_regressors(estimator=None, search_space=None, address_to_save=None, n_iter = 50):
    current_time = datetime.now().strftime("%H:%M:%S")
    print("Starting bayesian optimisation of regressors \n Time =", current_time)
    pkl_address = address_to_save + ".pkl"
    csv_address = address_to_save + ".csv"
    my_data = MyDataset()
    regressors = {}
    target_train = {}
    features_train = my_data.x_train
    features_test = my_data.x_test

    if not(os.path.isfile(pkl_address)):
        """
            For each of the 10 XGBoost parameters fit a regressor using BO 
        """
        for index, parameter in enumerate(hyperparameters_data):
            print("Evaluating regressor for: ", parameter)
            target_train[parameter] = my_data.y_train[:, index]
            cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
            search = BayesSearchCV(estimator=estimator, search_spaces=search_space, n_jobs=-1, cv=cv, n_iter=n_iter)
            # Performing Bayesian Optimisation
            search.fit(features_train.numpy(), target_train[parameter].numpy())
            # Reporting the best config for a parameters Regressor
            print(search.best_score_)
            print(search.best_params_)
            regressors[parameter] = {"best score": search.best_score_, "best config": search.best_params_}

        """Saving the configurations of the best performing models from Bayesian Optimisation"""
        with open(pkl_address, 'wb') as f:
            pickle.dump(regressors, f)

    with open(pkl_address, 'rb') as f:
        loaded_dict = pickle.load(f)
    print(loaded_dict)

    """Training the models and forming predictions"""
    models = {}
    predictions = pd.DataFrame()
    for index, parameter in enumerate(hyperparameters_data):
        target_train[parameter] = my_data.y_train[:, index]
        models[parameter] = estimator.__class__(**loaded_dict[parameter]['best config'])
        models[parameter].fit(features_train.numpy(), target_train[parameter].numpy())
        predictions[parameter] = models[parameter].predict(features_test)

    """Evaluating prediction quality"""
    baseline = pd.read_csv('./data/baseline_performance.csv')
    comparison = pd.DataFrame()
    for j in range(len(predictions)):
        """For each of the Test tasks (18 in total) we evaluate the performance of its predicted best config for XGBoost"""
        current_task = test_ids[j]
        print("Evaluating task with id: ", current_task)
        pred = predictions.iloc[j]
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
        """We convert the model predictions into a dictionary for the HPC"""
        new_config = {}
        for i in range(len(hyperparameters_data)):
            new_config[hyperparameters_data[i]] = pred[i]
        """Corrections to continuous predictions that might not be possible"""
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
    comparison.to_csv(csv_address)
    return comparison


if __name__ == "__main__":
    """
    compare_regressor_with_baseline - evaluations using fixed configs. Either a "good" default config found online or the default config was used
    """

    # Added to not overwrite any of the previous reults
    default_address = './data/bayes_results_'

    params_forest = {
        'n_estimators': 200,
        'min_samples_split': 2,
        'min_samples_leaf': 3,
        'max_features': 'sqrt',
        'max_depth': 120,
        'bootstrap': False
    }
    params_boosting = {
        "n_estimators": 500,
        "max_depth": 5,  # was 4
        "min_samples_split": 2,  # was 5
        "learning_rate": 0.01
        # "loss": "squared_error",
    }
    params_knn = {}
    params_xgboost = {}
    params_mlperceptron = {'batch_size': 1, 'early_stopping': True, 'validation_fraction': 0.1}

    print("Fitting Random forest regressor \n")
    random_forest_results = compare_regressor_with_baseline(regressor=ensemble.RandomForestRegressor(**params_forest))
    display(random_forest_results)
    # random_forest_results.to_csv("./data/random_forest_comparison.csv")

    print("Fitting Gradient boosting regressor\n")
    gradient_boosting_results = compare_regressor_with_baseline(
        regressor=ensemble.GradientBoostingRegressor(**params_boosting))
    display(gradient_boosting_results)
    # gradient_boosting_results.to_csv("./data/gradient_boosting_comparison.csv")

    print("Fitting FFN regressor \n")
    neural_net_results = compare_regressor_with_baseline(custom_model=True)
    display(neural_net_results)
    # neural_net_results.to_csv("./data/neural_net.csv")

    print("Fitting KNN regressor \n")
    knn_result = compare_regressor_with_baseline(regressor=KNeighborsRegressor(**params_knn))
    display(knn_result)
    # knn_result.to_csv("./data/kmeans_multitarget.csv")

    print("Fitting MLPerceptron regressor \n")
    multi_layer_perceptron_results = compare_regressor_with_baseline(regressor=MLPRegressor(**params_mlperceptron))
    display(multi_layer_perceptron_results)
    # multi_layer_perceptron_results.to_csv("./data/mlperceptron_default.csv")

    print("Fitting XGBoost regressor \n")
    xgboost_results = compare_regressor_with_baseline(regressor=XGBRegressor(**params_xgboost))
    display(xgboost_results)

    print("Fitting KNN regressors for different K \n")
    evaluate_knn_regressors([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15])


    """
    Here we use the evaluate_bayesian_regressors function to fit regressors using Bayesian Optimisation for each of the 10 XGBoost hyperparameters
    We do this for Random forest, XGBoost Regressor and KNNRegressor
    """
    print("Fitting models using Bayesian optimisation")
    my_data = MyDataset()
    features_train = my_data.x_train
    features_test = my_data.x_test
    target_train = {}

    # Search space for KNeighborsRegressor
    search_space = dict()
    search_space['n_neighbors'] = (1, 20)
    search_space['weights'] = ['uniform', 'distance']  # not necessary uniform always works
    search_space['leaf_size'] = (2, 60)
    search_space['algorithm'] = ['auto', 'ball_tree', 'kd_tree', 'brute']
    search_space['p'] = (1, 3)
    bayes_knn_result = evaluate_bayesian_regressors(estimator=KNeighborsRegressor, search_space=search_space,
                                                    address_to_save=default_address + "knn")
    display(bayes_knn_result)

    # Search space for RandomForestRegressor
    search_space = dict()
    search_space['n_estimators'] = (50, 500)
    search_space['max_features'] = ['auto', 'sqrt']
    search_space['max_depth'] = (5, 25)
    search_space['min_samples_split'] = [2, 5, 10]
    search_space['min_samples_leaf'] = [1, 2, 4]
    search_space['bootstrap'] = ['True', 'False']
    bayes_forest_result = evaluate_bayesian_regressors(estimator=ensemble.RandomForestRegressor,
                                                       search_space=search_space,
                                                       address_to_save=default_address + "rforest")
    display(bayes_forest_result)

    # MLPerceptron regression - DOESNT WORK, BayesSearchCV doesn't allow for the number of layers to be specified, and passing arrays/tupples is not allowed
    # search_space = dict()
    # search_space['hidden_layer_sizes'] = (1,51)
    # search_space["activation"] = ["identity", "logistic", "tanh", "relu"]
    # search_space["solver"] = ["lbfgs","sgd","adam"]
    # search_space["alpha"] = (0.00005, 0.0005, 'uniform')

    # Search space XGBoost
    search_space = dict()
    search_space['eta'] = Real(0, 1, 'uniform')
    search_space['n_estimators'] = Integer(50, 500)
    search_space['max_depth'] = Integer(3, 15)
    search_space['learning_rate'] = Real(10e-4, 0.1, 'log-uniform')
    search_space['colsample_bytree'] = Real(0.3, 0.9, 'uniform')
    search_space['subsample'] = Real(0.3, 1, 'uniform')
    bayes_xgb_result = evaluate_bayesian_regressors(estimator=XGBRegressor,
                                                    search_space=search_space,
                                                    address_to_save=default_address + "xgboost")
