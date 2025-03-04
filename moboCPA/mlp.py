import numpy as np
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold

tkwargs = {
    "dtype": torch.double, 
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(7,16, **tkwargs),
            nn.Sigmoid(),
            nn.Linear(16,32),
            nn.Sigmoid(),
            nn.Linear(32,16),
            nn.Sigmoid(),
            nn.Linear(16,1)
        )

    def forward(self, X):
        return self.layers(X)

    def fit(self, X_train, y_train, X_test=None, y_test=None, n_epochs=1000, lr=0.01, verbose=False, plot=False):
        # Switch to training mode
        self.train()
        # Initialize loss function and optimizer
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # Initialize lists to store loss
        if plot:
            train_loss_hist = []
            test_loss_hist = []
        # Training loop
        for epoch in range(n_epochs):
            # Forward pass
            y_pred_train = self(X_train).flatten()
            train_loss = loss_fn(y_pred_train, y_train)
            if plot:
                train_loss_hist.append(train_loss.item())
            # Backward pass
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            # Compute test loss if test data is provided
            if X_test is not None and y_test is not None:
                self.eval()
                with torch.no_grad():
                    y_pred_test = self(X_test).flatten()
                    test_loss = loss_fn(y_pred_test, y_test)
                    if plot:
                        test_loss_hist.append(test_loss.item())
                self.train()
            # Print loss
            if verbose and (epoch+1) % 100 == 0:
                if X_test is not None and y_test is not None:
                    print(f"Epoch {epoch+1}/{n_epochs}: Train Loss = {train_loss.item()} - Test Loss = {test_loss.item()}")
                else:
                    print(f"Epoch {epoch+1}/{n_epochs}: Train Loss = {train_loss.item()}")
        # Generate plot
        if plot:
            plt.plot(train_loss_hist, label="Train Loss")
            if X_test is not None and y_test is not None:
                plt.plot(test_loss_hist, label="Test Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()
        # Trained model evaluation
        self.eval()
        with torch.no_grad():
            y_pred_train = self(X_train)
            R2_train = r2_score(y_train.cpu().numpy(), y_pred_train.cpu().numpy())
            MSE_train = mean_squared_error(y_train.cpu().numpy(), y_pred_train.cpu().numpy())
            if X_test is not None and y_test is not None:
                y_pred_test = self(X_test)
                R2_test = r2_score(y_test.cpu().numpy(), y_pred_test.cpu().numpy())
                MSE_test = mean_squared_error(y_test.cpu().numpy(), y_pred_test.cpu().numpy())
                return R2_train, MSE_train, R2_test, MSE_test
            else:
                return R2_train, MSE_train

    def predict(self, X):
        # Switch to evaluation mode
        self.eval()
        with torch.no_grad():
            return self(X)