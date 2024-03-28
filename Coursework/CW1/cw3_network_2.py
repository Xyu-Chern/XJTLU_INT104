import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

file_path = 'Coursework/CW_Data.xlsx' 
data = pd.read_excel(file_path)
original_data=pd.read_excel(file_path)
programme_mapping = {
    1: [0, 0],
    2: [0, 1],
    3: [1, 0],
    4: [1, 1]
}
data['Programme'] = data['Programme'].apply(lambda x: programme_mapping[x])


data = data.drop(columns=['Index'])
features = data.drop(columns=['Programme','Total','Grade','Gender'])
labels = pd.DataFrame(data['Programme'].tolist(), columns=['Prog_1', 'Prog_2'])


scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

X_tensor = torch.tensor(features_scaled, dtype=torch.float)
y_tensor = torch.tensor(labels.values, dtype=torch.float)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(X_tensor.shape[1], 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, X_tensor.shape[1])
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def accuracy(y_pred, y_true):
    y_pred_labels = torch.argmax(y_pred, dim=1)
    y_true_labels = torch.argmax(y_true, dim=1)
    correct = (y_pred_labels == y_true_labels).sum().item()
    total = y_true.size(0)
    return correct / total

checkpoint = 1
if checkpoint ==1:
    losses = []
    accuracies = []
    num_epochs = 10000
    for epoch in range(num_epochs):
        outputs = model(X_tensor)
        p =criterion(model.encoder(X_tensor),y_tensor)
        loss = criterion(outputs,  X_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(loss)
        losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            acc = accuracy(outputs,  X_tensor)
            accuracies.append(acc)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {acc:.4f}, P:{p}')
            torch.save(model.state_dict(), 'Coursework/CW1/simple_nn_model_2.pth')

    colors = ['r', 'g', 'b', 'y']
    for i in range(len(original_data)):
        programme = int(original_data.iloc[i]['Programme']) - 1
        plt.scatter(model.encoder(X_tensor).detach().numpy()[i, 0], model.encoder(X_tensor).detach().numpy()[i, 1], c=colors[programme],  marker='o', edgecolor='k', s=50, alpha=0.6)

    plt.title('network visualization')
    plt.xlabel('comp 1')
    plt.ylabel('comp 2')
    plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), label='Cluster ID')
    plt.show()

    torch.save(model.state_dict(), 'Coursework/CW1/simple_nn_model_2.pth')

    plt.plot(losses, label='Loss')
    plt.plot(np.arange(9, num_epochs, 10), accuracies, label='Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

else:
    model.load_state_dict(torch.load('Coursework/CW1/simple_nn_model_2.pth'))

    colors = ['r', 'g', 'b', 'y']
    for i in range(len(original_data)):
        programme = int(original_data.iloc[i]['Programme']) - 1
        plt.scatter(model.encoder(X_tensor).detach().numpy()[i, 0], model.encoder(X_tensor).detach().numpy()[i, 1], c=colors[programme],  marker='o', edgecolor='k', s=50, alpha=0.6)

    plt.title('network visualization')
    plt.xlabel('comp 1')
    plt.ylabel('comp 2')
    plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), label='Cluster ID')
    plt.show()



