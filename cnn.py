from scipy.io import loadmat
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import statistics as stats
import matplotlib.pyplot as plt
import seaborn as sns

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

DATAFILE = "./deepsat4/sat-4-full.mat"  # https://csc.lsu.edu/~saikat/deepsat/
BATCH_SIZE = 128
LR = 0.001
EPOCHS = 100


class Data(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.Tensor(x_data).permute((2, 0, 1, 3)).to(DEVICE)

        # # Normalize the input data
        # self.x_data -= self.x_data.mean(dim=(0, 1, 2))
        # self.x_data /= self.x_data.std(dim=(0, 1, 2))

        # Standardize the data (per-channel min-max standardization)
        pc_min, pc_max = self.x_data.reshape(4, -1).min(dim=1).values, self.x_data.reshape(4, -1).max(dim=1).values
        for i in range(4):
            self.x_data[i] -= pc_min[i]
            self.x_data[i] /= pc_max[i] - pc_min[i]

        self.y_data = torch.Tensor(y_data).to(DEVICE)

    def __len__(self):
        return self.x_data.shape[-1]

    def __getitem__(self, i):
        return self.x_data[:, :, :, i], self.y_data[:, i]


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=5, stride=1, kernel_size=5)
        self.pool1 = nn.AvgPool2d(kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=12, stride=1, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=588, out_features=4)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.pool1(self.conv1(x)))
        x = self.relu(self.pool2(self.conv2(x)))
        x = self.fc(self.flatten(x))
        return self.softmax(x)


def train(cnn, dataloader, loss_func, optimizer):
    train_loss, train_accuracy = [], []

    cnn.train()
    for x, y in dataloader:
        # Zero gradients and compute the prediction
        optimizer.zero_grad()
        prediction = cnn(x)

        # Loss computation and backpropagation
        loss = loss_func(prediction, y)
        loss.backward()

        # Parameter optimization
        optimizer.step()

        # Track loss and accuracy metrics
        train_loss.append(loss.item())
        train_accuracy.append((torch.argmax(y, dim=1) == torch.argmax(prediction, dim=1)).sum().item() / len(y))

    return train_loss, train_accuracy


def test(cnn, dataloader, loss_func):
    test_loss, test_accuracy = [], []

    cnn.eval()
    for x, y in dataloader:
        # Obtain predictions and track loss and accuracy metrics
        prediction = cnn(x)
        test_loss.append(loss_func(prediction, y).item())
        test_accuracy.append((torch.argmax(y, dim=1) == torch.argmax(prediction, dim=1)).sum().item() / len(y))

    return test_loss, test_accuracy


def run(epochs=None, lr=None, batch_size=None):
    global EPOCHS, LR, BATCH_SIZE
    EPOCHS, LR, BATCH_SIZE = epochs or EPOCHS, lr or LR, batch_size or BATCH_SIZE

    print("Loading data...")
    # Load the DeepSat-4 dataset
    ntrain = 9000
    ntest = 1000
    data = loadmat(DATAFILE)
    x_train, x_test, y_train, y_test = (
        data["train_x"][:, :, :, :ntrain],
        data["test_x"][:, :, :, :ntest],
        data["train_y"][:, :ntrain],
        data["test_y"][:, :ntest],
    )

    # Define the datasets
    train_data = Data(x_train, y_train)
    test_data = Data(x_test, y_test)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # Instantiate the model
    cnn = CNN()
    cnn.to(DEVICE)

    # Define the optimizer and loss function
    optimizer = Adam(cnn.parameters(), lr=LR)
    categorical_cross_entropy = nn.CrossEntropyLoss()

    # Training loop
    try:
        print("Training CNN model...")
        train_loss, test_loss = [], []
        train_acc, test_acc = [], []
        for i in range(EPOCHS):
            loss, acc = train(cnn, train_loader, categorical_cross_entropy, optimizer)
            train_loss.append(stats.mean(loss))
            train_acc.append(stats.mean(acc))

            loss, acc = test(cnn, test_loader, categorical_cross_entropy)
            test_loss.append(stats.mean(loss))
            test_acc.append(stats.mean(acc))
            print(
                f'Epoch {i + 1}/{EPOCHS}  |  train loss {train_loss[-1]:.4f}  |  train acc {train_acc[-1]:.2%}  |  test loss {test_loss[-1]:.4f}  |  test acc {test_acc[-1]:.2%}'
            )
    except KeyboardInterrupt as e:
        if not test_acc:
            raise KeyboardInterrupt(e)

    if __name__ == '__main__':
        # Plot the results
        plt.figure()
        sns.lineplot(train_loss, label='train')
        sns.lineplot(test_loss, label='test')
        plt.title('Loss')

        plt.figure()
        sns.lineplot(train_acc, label='train')
        sns.lineplot(test_acc, label='test')
        plt.title('Accuracy')

        plt.show()

    print(test_acc)
    return train_acc, test_acc, train_loss, test_loss


if __name__ == '__main__':
    run()
