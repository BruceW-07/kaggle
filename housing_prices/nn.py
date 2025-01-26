# %% [markdown]
# ## 数据准备

# %%
import pandas as pd
import numpy as np

# %%
train_file_path = 'data/train.csv'
test_file_path = 'data/test.csv'

train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# %%
print(train_data.columns)

# %%
train_data_clean = train_data.dropna(axis=1)
test_data_clean = test_data.dropna(axis=1)

print(train_data_clean.columns)
print(test_data_clean.columns)

# %%
house_features = list(train_data_clean.select_dtypes(include=[np.number]).columns.drop(['Id', 'SalePrice']))
house_features = list(set(house_features).intersection(set(test_data_clean.columns)))

print(house_features)
print(len(house_features))

# %%
train_df = train_data_clean[house_features + ['SalePrice']]

test_df = test_data[house_features]

# %%
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# %%
class DataFrameDataset(Dataset):
    def __init__(self, df):
        self.df = df
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        features = torch.tensor(self.df.iloc[idx].values[:-1], dtype=torch.float32)
        label = torch.tensor(self.df.iloc[idx].values[-1], dtype=torch.float32)
        return features, label

# %%
train_dataset = DataFrameDataset(train_df)

batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for X, y in train_dataloader:
  print(f"Shape of X: {X.shape}")
  print(f"Shape of y: {y.shape}")
  break

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(len(house_features), 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        price = self.linear_relu_stack(x)
        return price
    
model = NeuralNetwork().to(device)
print(model)

# %%
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# %%
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        y = y.view(-1, 1)
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# %%
epochs = 20
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
print("Done!")

# %%
model.eval()
with torch.no_grad():

    train_X = torch.tensor(train_df.values[:, :-1], dtype=torch.float32).to(device)
    train_y = torch.tensor(train_df.values[:, -1], dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        train_pred = model(train_X[:10])
        for i in range(10):
            print(f"Predicted: {train_pred[i].item()}, Actual: {train_y[i].item()}")


# %%
model.eval()
with torch.no_grad():
    test_X = torch.tensor(test_df.values, dtype=torch.float32).to(device)
    test_pred = model(test_X)
    submission = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': test_pred.cpu().numpy().flatten()})
    submission.to_csv('submission.csv', index=False)

# %%
print(type(test_df))

# %%
print(test_data[test_data['Id'] == 2121][house_features])

# %%



