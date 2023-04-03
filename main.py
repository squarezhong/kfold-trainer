# file IO
import pandas as pd
# pytorch library
import torch
from model import FNN
from torch.utils.data import TensorDataset
from train import Trainer

# read csv file
fileName = 'ecg_data.csv'
data = pd.read_csv(fileName, header=None)
features = data.iloc[:,:-1] # take all except last column

# the original result labels are '1, 2, 3, 4', minus 1 we get '0, 1, 2, 3', they are convenient for computation
labels = data.iloc[:,-1] - 1 # take last column


ecg_dataset = TensorDataset(torch.tensor(features.values, dtype=torch.float32), torch.tensor(labels.values, dtype=torch.float32))

# instantiate the network and Trainer, then train and evaluate
# fully-connected net
full_connected_model = FNN(188, 4)
full_connected_trainer = Trainer(full_connected_model, ecg_dataset, lr=4e-4, n_epochs=200)
full_connected_trainer.train_evaluate('full_connected')
