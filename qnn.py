import pickle
import os
import statistics as stats

from scipy.io import loadmat
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import BallTree
import numpy as np

from quanvolution import Quanvolution
from constants import FILTERS

DATAFILE = "./deepsat4/sat-4-full.mat"  # https://csc.lsu.edu/~saikat/deepsat/
BATCH_SIZE = 128
LR = 0.001
EPOCHS = 10


class Data(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.Tensor(x_data).permute((2, 0, 1, 3))

        # # Normalize the input data
        # self.x_data -= self.x_data.mean(dim=(0, 1, 2))
        # self.x_data /= self.x_data.std(dim=(0, 1, 2))

        # Standardize the data
        mx, mn = self.x_data.max(), self.x_data.min()
        self.x_data -= mn
        self.x_data /= mx - mn

        self.y_data = torch.Tensor(y_data)

    def __len__(self):
        return self.x_data.shape[-1]

    def __getitem__(self, i):
        return self.x_data[:, :, :, i], self.y_data[:, i]


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv2 = nn.Conv2d(in_channels=5, out_channels=12, stride=1, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=1452, out_features=4)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(x)  # Input already has the quanvolutional layer applied
        x = self.relu(self.pool2(self.conv2(x)))
        x = self.fc(self.flatten(x))
        return self.softmax(x)


def prerun_quanvolution():
    train_loader, _ = load_data()
    quanv = Quanvolution(nfilters=5, kernel_size=5, manual_filters=FILTERS, max_cores=6)
    kernel_size = 5
    block_expectation_pairs = {}

    n = 0
    try:
        for img_batch, _ in train_loader:
            img_batch = torch.mean(img_batch, dim=1, keepdim=True)  # Average out the channels

            img_blocks = img_batch.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
            img_blocks = img_blocks.reshape(-1, kernel_size, kernel_size)

            for block in img_blocks:
                n += 1
                if n < 0:
                    continue
                print(n)
                expectations = quanv(block)
                block_expectation_pairs[block] = expectations
                if n % 1000 == 0:
                    print(f'Blocks processed: {n}')
                    write_pickle_file(block_expectation_pairs)
                    block_expectation_pairs = {}

    except:
        print(f'Interrupted at {n}/{9000 * 24 * 24} blocks.')

    write_pickle_file(block_expectation_pairs)


def write_pickle_file(block_expectation_pairs, dir='processed_blocks'):
    file_nums = [int(f[:-4]) for f in os.listdir(dir) if f[:-4].isnumeric()]
    n = max(file_nums) if file_nums else -1
    file = os.path.join(dir, f'{n+1}.pkl')
    print(f'Writing \'{file}\'')
    with open(file, 'wb') as f:
        pickle.dump(block_expectation_pairs, f)


def define_balltree_from_pickle_files(directory='processed_blocks'):
    all_blocks = {}
    for file in os.listdir(directory):
        if not file[:-4].isnumeric():
            continue
        with open(os.path.join(directory, file), 'rb') as f:
            block_expectation_pairs = pickle.load(f)

        all_blocks |= block_expectation_pairs

    blocks_flattened_numpy = np.array([x.flatten().numpy() for x in list(all_blocks.keys())])
    all_blocks = {tuple(x.flatten().numpy()): y for x, y in all_blocks.items()}
    balltree = BallTree(blocks_flattened_numpy)
    return all_blocks, balltree


def load_data(ntrain=9000, ntest=1000):
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

    return train_loader, test_loader


def apply_quanv(t, balltree, block_expectation_pairs, kernel_size):
    t = torch.mean(t, dim=1, keepdim=True)  # Must keep the dimension so that torch.unfold works properly

    bs = t.shape[0]
    nfilters = len(list(block_expectation_pairs.values())[0])
    ks2 = kernel_size**2
    iout = t.shape[2] - kernel_size + 1
    jout = t.shape[3] - kernel_size + 1
    # Output tensor has shape (bs, nfilters, iout, jout)

    # # Unfold the input tensor to obtain all blocks on which the quanvolution operation operates
    # t_blocks = t.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
    # t_blocks = t_blocks.reshape(-1, kernel_size, kernel_size)

    out = torch.empty([bs, nfilters, iout, jout])
    for batch_index in range(bs):
        for i in range(iout):
            for j in range(jout):
                block = t.squeeze(1)[batch_index, i : i + kernel_size, j : j + kernel_size].reshape(1, ks2).numpy()
                index = balltree.query(block, return_distance=False)[0][0]
                closest_processed_block = tuple(balltree.get_arrays()[0][index])
                expectation_values = block_expectation_pairs[closest_processed_block]

                out[batch_index, :, i, j] = torch.Tensor(expectation_values)
    return out


def train(cnn, dataloader, loss_func, optimizer, balltree, block_expectation_pairs):
    train_loss, train_accuracy = [], []

    cnn.train()
    for x, y in dataloader:
        x = apply_quanv(x, balltree, block_expectation_pairs, 5)

        # Zero gradients and compute the prediction
        optimizer.zero_grad()
        prediction = cnn(x)

        # Loss computation and backpropagation
        loss = loss_func(prediction, y)
        loss.backward()

        # Weight optimization
        optimizer.step()

        # Track loss and acuracy metrics
        train_loss.append(loss.item())
        train_accuracy.append((torch.argmax(y, dim=1) == torch.argmax(prediction, dim=1)).sum().item() / len(y))

    return train_loss, train_accuracy


def test(cnn, dataloader, loss_func, balltree, block_expectation_pairs):
    test_loss, test_accuracy = [], []

    cnn.eval()
    for x, y in dataloader:
        x = apply_quanv(x, balltree, block_expectation_pairs, 5)
        # Obtain predictions and track loss and accuracy metrics
        prediction = cnn(x)
        test_loss.append(loss_func(prediction, y).item())
        test_accuracy.append((torch.argmax(y, dim=1) == torch.argmax(prediction, dim=1)).sum().item() / len(y))

    return test_loss, test_accuracy


def main():
    print("Loading data...")
    # Load the DeepSat-4 dataset
    train_loader, test_loader = load_data()

    # Load the data from the pre-run convolution operation
    block_expectation_pairs, balltree = define_balltree_from_pickle_files()

    # Instantiate the model
    cnn = CNN()

    # Define the optimizer and loss function
    optimizer = Adam(cnn.parameters(), lr=LR)
    categorical_cross_entropy = nn.CrossEntropyLoss()

    # Training loop
    try:
        print("Training CNN model...")
        train_loss, test_loss = [], []
        train_acc, test_acc = [], []
        for i in range(EPOCHS):
            loss, acc = train(
                cnn, train_loader, categorical_cross_entropy, optimizer, balltree, block_expectation_pairs
            )
            train_loss.append(stats.mean(loss))
            train_acc.append(stats.mean(acc))

            loss, acc = test(cnn, test_loader, categorical_cross_entropy, balltree, block_expectation_pairs)
            test_loss.append(stats.mean(loss))
            test_acc.append(stats.mean(acc))
            print(
                f"Epoch {i + 1}/{EPOCHS}  |  train loss {train_loss[-1]:.4f}  |  train acc {train_acc[-1]:.2%}  |  test loss {test_loss[-1]:.4f}  |  test acc {test_acc[-1]:.2%}"
            )
    except KeyboardInterrupt as e:
        if not test_acc:
            raise KeyboardInterrupt(e)

    # Plot the results
    plt.figure()
    sns.lineplot(train_loss, label="train")
    sns.lineplot(test_loss, label="test")
    plt.title("Loss")

    plt.figure()
    sns.lineplot(train_acc, label="train")
    sns.lineplot(test_acc, label="test")
    plt.title("Accuracy")

    plt.show()


if __name__ == "__main__":
    # main()
    prerun_quanvolution()