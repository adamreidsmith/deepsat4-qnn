from scipy.io import loadmat
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
# import qiskit
import statistics as stats
import matplotlib.pyplot as plt
import seaborn as sns

BATCH_SIZE = 256
LR = 0.001
EPOCHS = 100

print('Loading data...')
# Load the DeepSat-4 dataset: https://csc.lsu.edu/~saikat/deepsat/
datafile = './deepsat4/sat-4-full.mat'
data = loadmat(datafile)
x_train, x_test, y_train, y_test = data['train_x'], data['test_x'], data['train_y'], data['test_y']


class Data(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.Tensor(x_data).permute((2, 0, 1, 3))

        # Normalize the input data
        self.x_data -= self.x_data.mean(dim=(0, 1, 2))
        self.x_data /= self.x_data.std(dim=(0, 1, 2))

        self.y_data = torch.Tensor(y_data)

    def __len__(self):
        return self.x_data.shape[-1]

    def __getitem__(self, i):
        return self.x_data[:, :, :, i], self.y_data[:, i]


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=4, out_channels=5, stride=1, kernel_size=5)
        self.pool1 = nn.AvgPool2d(kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=5, out_channels=12, stride=1, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=588, out_features=4)
        self.relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.pool1(self.conv1(x)))
        x = self.relu(self.pool2(self.conv2(x)))
        x = self.fc(self.flatten(x))
        return self.softmax(x)


# Define the datasets
train_data = Data(x_train, y_train)
test_data = Data(x_test, y_test)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# Instantiate the model
cnn = CNN()

# Define the optimizer and loss function
optimizer = Adam(cnn.parameters(), lr=LR)
categorical_cross_entropy = nn.CrossEntropyLoss()


def train():
    train_loss, train_accuracy = [], []

    cnn.train()
    for x, y in train_loader:
        # Zero gradients and compute the prediction
        optimizer.zero_grad()
        prediction = cnn(x)

        # Loss computation and backpropagation
        loss = categorical_cross_entropy(prediction, y)
        loss.backward()

        # Weight optimization
        optimizer.step()

        train_loss.append(loss.item())

        train_accuracy.append((torch.argmax(y, dim=1) == torch.argmax(
            prediction, dim=1)).sum().item() / len(y))

    return train_loss, train_accuracy


def test():
    test_loss, test_accuracy = [], []

    cnn.eval()
    for x, y in test_loader:
        prediction = cnn(x)
        test_loss.append(categorical_cross_entropy(prediction, y).item())

        test_accuracy.append((torch.argmax(y, dim=1) == torch.argmax(
            prediction, dim=1)).sum().item() / len(y))

    return test_loss, test_accuracy


try:
    print('Training CNN model...')
    train_loss, test_loss = [], []
    train_acc, test_acc = [], []
    for i in range(EPOCHS):
        loss, acc = train()
        train_loss.append(stats.mean(loss))
        train_acc.append(stats.mean(acc))

        loss, acc = test()
        test_loss.append(stats.mean(loss))
        test_acc.append(stats.mean(acc))
        print(
            f'Epoch {i + 1} / {EPOCHS}  |  train loss {train_loss[-1]:.4f}  |  train acc {train_acc[-1]:.2%}  |  test loss {test_loss[-1]:.4f}  |  test acc {test_acc[-1]:.2%}')
except KeyboardInterrupt as e:
    if not test_acc:
        raise KeyboardInterrupt(e)

plt.figure()
sns.lineplot(train_loss, label='train')
sns.lineplot(test_loss, label='test')
plt.title('Loss')

plt.figure()
sns.lineplot(train_acc, label='train')
sns.lineplot(test_acc, label='test')
plt.title('Accuracy')

plt.show()
