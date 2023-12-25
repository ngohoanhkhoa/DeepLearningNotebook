"""
This is a boilerplate pipeline 'pipeline_digits'
generated using Kedro 0.18.13
"""
from typing import Tuple
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

def get_dataset() -> pd.DataFrame:
    data = load_digits()
    df_data = pd.DataFrame(data['data'], columns=data['feature_names'])
    df_data['target'] = data['target']
    return df_data

def split_dataset(df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame]:
    y = df.pop('target')
    X = df

    df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(X, y, test_size=test_size)

    return df_X_train, df_X_test, df_y_train, df_y_test

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_1 = nn.Linear(64, 32)
        self.linear_2 = nn.Linear(32, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        x_prob = self.softmax(x)
        return x_prob

def fit_model(df_X: pd.DataFrame, df_y: pd.DataFrame, epochs: int, batch_size: int) -> nn.Module:
    model = NeuralNetwork()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.002)
    tensor_X = torch.from_numpy(df_X.to_numpy()).float()
    tensor_y = torch.from_numpy(df_y.to_numpy().squeeze())

    dataloader = DataLoader(
        TensorDataset(tensor_X, tensor_y),
        batch_size=batch_size,
        shuffle=True
    )

    for epoch in range(epochs):
        for x, y in dataloader:
            optimizer.zero_grad()
            y_prob = model.forward(x)
            loss = nn.NLLLoss()(torch.log(y_prob), y)
            loss.backward()
            optimizer.step()
        
    return model

def predict_data(model: nn.Module, df_X: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    model = NeuralNetwork()
    tensor_X = torch.from_numpy(df_X.to_numpy()).float()

    dataloader = DataLoader(
        tensor_X,
        batch_size=batch_size,
    )
    
    y_predict = []
    for x in dataloader:
        y_prob = model.forward(x)
        y_predict += list(torch.argmax(y_prob, dim=1).detach().numpy())

    df_y_predict = pd.DataFrame(y_predict, index=df_X.index, columns=['predict'])
    
    return df_y_predict

def evaluate_model(df_y_test: pd.DataFrame, df_y_predict: pd.DataFrame) -> pd.DataFrame:
    return df_y_test