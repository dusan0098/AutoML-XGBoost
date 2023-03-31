"""
    This file contains two classes:
    SimpleFFN
        - a Fully connected network for predicting XHBoost HPCs based on dataset features
        - the main function was used to manually test out different configurations and observe the CV scores
        - THIS MODEL WAS LATER NOT USED DUE TO NOT BEING COMPATIBLE WITH THE BAYESIAN OPTIMISATION FRAMEWORK
    MyDataset
        - originally meant to prepare data for the SimpleFFN
        - In the end was used for preparing data for all regressors that directly modelled the desired hyperparameter values (instead of the expected AUC)
"""

import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from project_utils import get_task_metafeatures, load_test_data, meta_feature_names, training_meta_features, \
    hyperparameters_data, default_config, test_ids, get_best_config_per_task, dataset_to_task, get_test_metafeatures, \
    get_all_metafeatures, get_dataset_to_task
from IPython.display import display


class SimpleFFN(nn.Module):
    """
    Fully connected NN implementation with variable number of layers, nodes per layer, activation function, dropout rate
  """

    def __init__(self, in_features=10, out_features=10, hidden_size=16, num_layers=4, activation='tanh', dropout=0.1):
        super().__init__()
        # Input layer with activation function and dropout
        input_layer = nn.Linear(in_features=in_features, out_features=hidden_size)
        layerlist = [input_layer, self.get_activation_layer(activation), nn.Dropout(dropout)]

        # Hidden layers
        for i in range(num_layers - 2):
            layerlist.append(nn.Linear(hidden_size, hidden_size))
            layerlist.append(self.get_activation_layer(activation))
            layerlist.append(nn.Dropout(dropout))

        # Output layer
        output_layer = nn.Linear(in_features=hidden_size, out_features=out_features)
        layerlist.append(output_layer)

        #Full architecture
        self.layers = nn.Sequential(*layerlist)

    def forward(self, x):
        """Forward pass"""
        return self.layers(x)

    @staticmethod
    def get_activation_layer(activation: str):
        if activation=="relu":
            return nn.ReLU()
        elif activation == "relu":
            return nn.Tanh()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        else:
            return nn.ReLU()

    def fit(self,x_train,y_train, loss_function ,num_epochs =20):
        self.apply(reset_weights)
        """Should be passed as arguments"""
        train_data = torch.utils.data.TensorDataset(x_train, y_train)
        trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=2)
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-3)

        for epoch in range(0, num_epochs):
            print(f'Starting epoch {epoch + 1}')
            current_loss = 0.0

            # Iterate over the DataLoader for training data
            for i, data in enumerate(trainloader, 0):
                # Get inputs
                inputs, targets = data
                # Zero the gradients
                optimizer.zero_grad()
                # Perform forward pass
                outputs = self(inputs)
                # Compute loss
                loss = loss_function(outputs, targets)
                # Perform backward pass
                loss.backward()
                # Perform optimization
                optimizer.step()
                # Print statistics
                current_loss += loss.item()
                if i % 10 == 9:
                    print('Loss after mini-batch %5d: %.3f' %
                          (i + 1, current_loss / 10))
                    current_loss = 0.0
            print('Training process has finished.')



"""
    Generalised RMSELoss where each dimension has a weight
"""
class RMSELoss(torch.nn.Module):
    def __init__(self, weights):
        super(RMSELoss, self).__init__()
        self.weights = weights

    def forward(self, x, y):
        criterion = nn.MSELoss()
        new_loss = torch.sqrt(criterion(torch.div(x, self.weights), torch.div(y, self.weights)))
        return new_loss

"""
    Helper function used to reset model in each CV fold 
"""
def reset_weights(model):
    '''
      Try resetting model weights to avoid
      weight leakage.
    '''
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

"""
    Class for preparing the training and test data for any model that predicts XGBoost hyperparameters directly
"""
class MyDataset(Dataset):
    def __init__(self):
        train_data, test_data = get_all_metafeatures(impute=True, best_configs=True)

        x = train_data[training_meta_features].values
        y = train_data[hyperparameters_data].values
        x_test = test_data[training_meta_features].values

        """
        x_train - Features of datasets in trainign set (94 rows)
        y_train - Best XGBoost configuration for each of the training datasets
        x_test - Relevant features of test datasets (18) for predicting best config 
        """
        self.x_train = torch.tensor(x, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.float32)
        self.train = torch.utils.data.TensorDataset(self.x_train, self.y_train)

        #Test data
        self.x_test =torch.tensor(x_test, dtype=torch.float32)

        """
        We also keep the ranges of each variable to use as weights in our models
        euclidian loss
        """
        self.x_range = torch.sub(torch.max(self.x_train, 0)[0], torch.min(self.x_train, 0)[0])
        self.y_range = torch.sub(torch.max(self.y_train, 0)[0], torch.min(self.y_train, 0)[0])

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]


"""Remove later"""
if __name__ == "__main__":

    # Additional parameters for k-fold CV
    torch.manual_seed(42)
    k_folds = 10

    # Get training data and variable ranges
    my_data = MyDataset()
    x_train = my_data.x_train
    y_train = my_data.y_train

    folds = KFold(n_splits=k_folds, shuffle=True)
    # New loss function defined for the model
    loss_function = RMSELoss(my_data.y_range)
    num_epochs = 20

    results = {}
    # Start print
    print('--------------------------------')

    # K-fold Cross Validation model evaluation
    for fold, (train_index, test_index) in enumerate(folds.split(x_train, y_train)):
        # Print
        print(f'FOLD {fold + 1}')
        print('--------------------------------')

        # Training data for this iteration
        x_train_fold = x_train[train_index]
        y_train_fold = y_train[train_index]
        train_data = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
        # Testing data for this iteration
        x_test_fold = x_train[test_index]
        y_test_fold = y_train[test_index]
        test_data = torch.utils.data.TensorDataset(x_test_fold, y_test_fold)

        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=2)

        testloader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=2)

        # Init the neural network
        network = SimpleFFN(in_features=x_train.size(dim=1), out_features=y_train.size(dim=1), hidden_size=16, num_layers=4, activation='tanh', dropout=0.1)
        network.apply(reset_weights)
        """
        Tried out different learning rates, after 5e-3 the CV loss indicates overfitting 
        for the default model (2 layer, hidden = 16, activation = ReLu)
        """
        optimizer = torch.optim.Adam(network.parameters(), lr=5e-3)

        # Run the training loop for defined number of epochs
        for epoch in range(0, num_epochs):
            print(f'Starting epoch {epoch + 1}')
            current_loss = 0.0

            # Iterate over the DataLoader for training data
            for i, data in enumerate(trainloader, 0):
                # Get inputs
                inputs, targets = data
                # Zero the gradients
                optimizer.zero_grad()
                # Perform forward pass
                outputs = network(inputs)
                # Compute loss
                loss = loss_function(outputs, targets)
                # Perform backward pass
                loss.backward()
                # Perform optimization
                optimizer.step()
                # Print statistics
                current_loss += loss.item()
                if i % 10 == 9:
                    print('Loss after mini-batch %5d: %.3f' %
                          (i + 1, current_loss / 10))
                    current_loss = 0.0

            print('Training process has finished.')

            print('Starting testing')

            # Evaluation for this fold
            total, full_loss = 0, 0
            with torch.no_grad():

                # Iterate over the test data and generate predictions
                for i, data in enumerate(testloader, 0):
                    # Get inputs
                    inputs, targets = data
                    # Generate outputs
                    outputs = network(inputs)
                    loss = loss_function(outputs, targets)
                    total += targets.size(0)
                    full_loss += loss

                # Print accuracy
                print('Accuracy for fold %d: %d ' % (fold, 100.0 * full_loss / total))
                print('--------------------------------')
                results[fold] = 100.0 * (full_loss / total)

            # Print fold results
        print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
        print('--------------------------------')
        sum = 0.0
        for key, value in results.items():
            print(f'Fold {key}: {value} ')
            sum += value
        print(f'Average CV loss: {sum / len(results.items())} ')
