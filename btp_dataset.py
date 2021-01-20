import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class BtpDataset(Dataset):
    """Btp time series dataset."""
    def __init__(self, csv_file, seq_len, normalize=True):
        """
        Args:
            csv_file (string): path to csv file
            normalize (bool): whether to normalize the data in [-1,1]
        """
        df = pd.read_csv(csv_file, sep=",")
        df['Timestamp'] = pd.to_datetime(df["data_column"].map(str) + " " + df["orario_column"], dayfirst=True)
        df = df.drop(['data_column', 'orario_column'], axis=1).set_index("Timestamp")
        data = df.to_numpy()
        # (batch_size, seq_len, num_of_features)
        # thay data cua pm vao.
        pm_dataset = pd.read_csv('./pm.csv')
        pm_dataset = pm_dataset.replace("**", 0)
        pm_dataset = pm_dataset.to_numpy()
        pm_data = pm_dataset[:, 4:5]
        pm_data = pm_data.astype(np.float)
        # data_for_lstm = np.empty(shape=(pm_data.shape[0]-seq_len, seq_len, pm_data.shape[1]-1))
        # for i in range(pm_data.shape[0]-seq_len):
        #     data_for_lstm[i, :, :] = pm_data[i:i+seq_len, 1:]
        # data_for_lstm = torch.from_numpy(data_for_lstm).float()
        # self.data = self.normalize(data_for_lstm) if normalize else data_for_lstm

        data_for_lstm = np.empty(shape=(data.shape[0]-seq_len, seq_len, data.shape[1]-1))
        for i in range(data.shape[0]-seq_len):
            data_for_lstm[i, :, :] = data[i:i+seq_len, 1:]
        data_for_lstm = torch.from_numpy(data_for_lstm).float()
        self.data = self.normalize(data_for_lstm) if normalize else data_for_lstm
        
        #Estimates distribution parameters of deltas (Gaussian) from normalized data
        original_deltas = data[:, -1] - data[:, 0]
        self.original_deltas = original_deltas
        self.or_delta_max, self.or_delta_min = original_deltas.max(), original_deltas.min() 
        deltas = self.data[:, -1] - self.data[:, 0]
        self.deltas = deltas
        self.delta_mean, self.delta_std = deltas.mean(), deltas.std()
        self.delta_max, self.delta_min = deltas.max(), deltas.min()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def normalize(self, x):
        """Normalize input in [-1,1] range, saving statics for denormalization"""
        self.max = x.max()
        self.min = x.min()
        return (2 * (x - x.min())/(x.max() - x.min()) - 1)
    
    def denormalize(self, x):
        """Revert [-1,1] normalization"""
        if not hasattr(self, 'max') or not hasattr(self, 'min'):
            raise Exception("You are calling denormalize, but the input was not normalized")
        return 0.5 * (x*self.max - x*self.min + self.max + self.min)
    
    def sample_deltas(self, number):
        """Sample a vector of (number) deltas from the fitted Gaussian"""
        return (torch.randn(number, 1) + self.delta_mean) * self.delta_std
    
    def normalize_deltas(self, x):
        return ((self.delta_max - self.delta_min) * (x - self.or_delta_min)/(self.or_delta_max - self.or_delta_min) + self.delta_min)
    
