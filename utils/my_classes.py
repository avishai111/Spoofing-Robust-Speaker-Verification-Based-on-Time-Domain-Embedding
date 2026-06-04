from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from sklearn.utils import shuffle

# This class is used to create a dataset object that can be used in the dataloader
@dataclass
class dataset(Dataset):
    data : np.ndarray 
    is_spoofed : pd.core.series.Series 
    chosen_labels_numeric : pd.core.series.Series
    attack_logical : pd.core.series.Series
    name : pd.core.series.Series
    speaker_id :  pd.core.series.Series
    data_transform : torch.Tensor = None
    labels_transform : torch.Tensor = None
    sex : pd.core.series.Series = None
    labels_format : str = 'is_spoofed' #['is_spoofed', 'one-hot','is_not_spoofed']
    data_for_gender_classification : np.ndarray = None
    data_without_separation  : np.ndarray = None
    
    def __len__(self):
        labels = self.is_spoofed
        return len(labels)
    
    def len_is_spoofed(self):
        return(self.is_spoofed.value_counts())
    
    def __getitem__(self, idx):
        data_sample = self.data[idx,:]  
        label_sample = (self.is_spoofed.iloc[idx].astype('uint8')).astype('float32') # is spoofed
        
        if self.labels_format == 'one-hot':
            label_sample = np.array([0,1] if label_sample == 0 else [1,0]) #one hot encoding
            
        if self.labels_format == 'is_not_spoofed':
            label_sample = np.array(1 if label_sample == 0 else 0) #convert 0 to 1 and 1 to 0 - is not spoofed
        
        if self.data_transform:
            data_sample = self.transform(data_sample)
        if self.labels_transform:
            label_sample = self.transform(label_sample)
        return data_sample,label_sample 
    
    def set_labels_format(self, labels_format):
        if labels_format not in ['is_spoofed', 'one-hot','is_not_spoofed']:
            raise ValueError("labels_format must be 'is_spoofed' or 'one-hot' or 'is_not_spoofed' .")
        self.labels_format = labels_format
    
    def shuffle(self,res = None):
        if res is None: # if we dont reasampling to the data and labels
            idx = np.random.permutation(len(self))
            self.data = self.data[idx,:]
            self.is_spoofed = self.is_spoofed.loc[idx]
            self.is_spoofed = self.is_spoofed.reset_index(drop=True)
            self.chosen_labels_numeric = self.chosen_labels_numeric.loc[idx]
            self.chosen_labels_numeric = self.chosen_labels_numeric.reset_index(drop=True)
            self.attack_logical = self.attack_logical.loc[idx]
            self.attack_logical = self.attack_logical.reset_index(drop=True)
            self.name = self.name.loc[idx]
            self.name = self.name.reset_index(drop=True)
            self.speaker_id = self.speaker_id.loc[idx]
            self.speaker_id = self.speaker_id.reset_index(drop=True)
            self.sex = self.sex.loc[idx]
            self.sex = self.sex.reset_index(drop=True)
            return self
        else: # if we do reasampling to the data and labels
            idx = np.random.permutation(len(self))
            self.data = self.data[idx,:]
            self.is_spoofed = self.is_spoofed.loc[idx]
            return self
        
        
        
        
        
    ##############################################################################################################
@dataclass    
class dataset_for_samo_loss(Dataset):
    data : np.ndarray 
    is_spoofed : pd.core.series.Series 
    chosen_labels_numeric : pd.core.series.Series
    attack_logical : pd.core.series.Series
    name : pd.core.series.Series
    speaker_id :  pd.core.series.Series
    data_transform : torch.Tensor = None
    labels_transform : torch.Tensor = None
    sex : pd.core.series.Series = None
    labels_format : str = 'is_spoofed' #['is_spoofed', 'one-hot','is_not_spoofed']
    
    def __len__(self):
        labels = self.is_spoofed
        return len(labels)
    
    def len_is_spoofed(self):
        return(self.is_spoofed.value_counts())
    
    def __getitem__(self, idx):
        data_sample = self.data[idx,:] 
       # attack_sample = self.attack_logical.iloc[idx]
        speaker_id_sample = self.speaker_id.iloc[idx]
      #  chosen_labels_numeric = self.chosen_labels_numeric.iloc[idx]
        label_sample = (self.is_spoofed.iloc[idx].astype('uint8')).astype('float32') # is spoofed
        
        return data_sample,label_sample,speaker_id_sample
    
    def set_labels_format(self, labels_format):
        if labels_format not in ['is_spoofed', 'one-hot','is_not_spoofed']:
            raise ValueError("labels_format must be 'is_spoofed' or 'one-hot' or 'is_not_spoofed' .")
        self.labels_format = labels_format
    
    def shuffle(self,res = None):
        if res is None: # if we dont reasampling to the data and labels
            idx = np.random.permutation(len(self))
            self.data = self.data[idx,:]
            self.is_spoofed = self.is_spoofed.loc[idx]
            self.is_spoofed = self.is_spoofed.reset_index(drop=True)
            self.chosen_labels_numeric = self.chosen_labels_numeric.loc[idx]
            self.chosen_labels_numeric = self.chosen_labels_numeric.reset_index(drop=True)
            self.attack_logical = self.attack_logical.loc[idx]
            self.attack_logical = self.attack_logical.reset_index(drop=True)
            self.name = self.name.loc[idx]
            self.name = self.name.reset_index(drop=True)
            self.speaker_id = self.speaker_id.loc[idx]
            self.speaker_id = self.speaker_id.reset_index(drop=True)
            self.sex = self.sex.loc[idx]
            self.sex = self.sex.reset_index(drop=True)
            return self
        else: # if we do reasampling to the data and labels
            idx = np.random.permutation(len(self))
            self.data = self.data[idx,:]
            self.is_spoofed = self.is_spoofed.loc[idx]
            return self
    
    
##############################################################################################################
from torch.utils.data import Dataset, DataLoader, Sampler
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
import random

class ImbalancedBatchSampler(Sampler):
    def __init__(self, spoof_indices, bonafide_indices, batch_size, spoof_ratio=0.9):
        self.spoof_indices = spoof_indices
        self.bonafide_indices = bonafide_indices
        self.batch_size = batch_size
        self.spoof_count = int(batch_size * spoof_ratio)
        self.bonafide_count = batch_size - self.spoof_count

    def __iter__(self):
        spoof_indices = random.sample(self.spoof_indices, len(self.spoof_indices))
        bonafide_indices = random.sample(self.bonafide_indices, len(self.bonafide_indices))
        
        for i in range(0, min(len(spoof_indices), len(bonafide_indices)), self.batch_size):
            batch_spoof = spoof_indices[i:i+self.spoof_count]
            batch_bonafide = bonafide_indices[i:i+self.bonafide_count]
            yield batch_spoof + batch_bonafide

    def __len__(self):
        return min(len(self.spoof_indices) // self.spoof_count, len(self.bonafide_indices) // self.bonafide_count)

@dataclass
class CustomDataset(Dataset):
    data: np.ndarray
    is_spoofed: pd.Series
    chosen_labels_numeric: pd.Series
    attack_logical: pd.Series
    name: pd.Series
    speaker_id: pd.Series
    data_transform: torch.Tensor = None
    labels_transform: torch.Tensor = None
    sex: pd.Series = None
    labels_format: str = 'is_not_spoofed'
    data_for_gender_classification: np.ndarray = None
    data_without_separation: np.ndarray = None

    def __len__(self):
        return len(self.is_spoofed)

    def __getitem__(self, idx):
        data_sample = self.data[idx, :]
        label_sample = float(self.is_spoofed.iloc[idx])
        self.attack_logical = self.attack_logical.loc[idx]
        self.attack_logical = self.attack_logical.reset_index(drop=True)
        self.speaker_id = self.speaker_id.loc[idx]
        self.speaker_id = self.speaker_id.reset_index(drop=True)
        self.sex = self.sex.loc[idx]
        self.sex = self.sex.reset_index(drop=True)
        
        if self.labels_format == 'one-hot':
            label_sample = np.array([0, 1] if label_sample == 0 else [1, 0])
        elif self.labels_format == 'is_not_spoofed':
            label_sample = 1 if label_sample == 0 else 0
        
        if self.data_transform:
            data_sample = self.data_transform(data_sample)
        if self.labels_transform:
            label_sample = self.labels_transform(label_sample)
        return data_sample, label_sample , self.attack_logical , self.speaker , self.sex

    def create_balanced_loader(self, batch_size=32, spoof_ratio=0.7):
        spoof_indices = self.is_spoofed[self.is_spoofed == 1].index.tolist()
        bonafide_indices = self.is_spoofed[self.is_spoofed == 0].index.tolist()
        sampler = ImbalancedBatchSampler(spoof_indices, bonafide_indices, batch_size, spoof_ratio)
        return DataLoader(self, batch_sampler=sampler)

    def set_labels_format(self, labels_format):
        if labels_format not in ['is_spoofed', 'one-hot', 'is_not_spoofed']:
            raise ValueError("labels_format must be 'is_spoofed', 'one-hot', or 'is_not_spoofed'.")
        self.labels_format = labels_format
