import pickle
import os
import statistics as stats
from itertools import chain
from multiprocessing import Pool
from functools import partial

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

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

DATAFILE = './deepsat4/sat-4-full.mat'  # https://csc.lsu.edu/~saikat/deepsat/
BATCH_SIZE = 2
LR = 0.001
EPOCHS = 100


class Data(Dataset):
    '''Class to represent the dataset'''

    def __init__(self, x_data, y_data):
        self.x_data = torch.Tensor(x_data).permute((2, 0, 1, 3))

        # Standardize the data (per-channel min-max standardization)
        pc_min, pc_max = self.x_data.reshape(4, -1).min(dim=1).values, self.x_data.reshape(4, -1).max(dim=1).values
        for i in range(4):
            self.x_data[i] -= pc_min[1]
            self.x_data[i] /= pc_max[i] - pc_min[i]

        self.y_data = torch.Tensor(y_data).to(DEVICE)

    def __len__(self):
        return self.x_data.shape[-1]

    def __getitem__(self, i):
        return self.x_data[:, :, :, i], self.y_data[:, i]


class QNN(nn.Module):
    '''Represents the neural network after the quanvolution filter has been applied.
    The filter has no trainable weights, so it does not need to be included in the optimization.
    '''

    def __init__(self):
        super().__init__()

        # self.pool1 = nn.AvgPool2d(kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=12, stride=1, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=1452, out_features=4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(x)  # Input already has the quanvolutional layer applied
        x = self.relu(self.pool2(self.conv2(x)))
        x = self.fc(self.flatten(x))
        return x


def prerun_quanvolution(start=0, max_cores=5):
    '''Runs the quanvolution operation in advance and saves the results in binary files'''

    train_loader, _ = load_data()
    quanv = Quanvolution(nfilters=5, kernel_size=5, manual_filters=FILTERS, max_cores=max_cores)
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
                if n < start:
                    continue
                print(n)
                expectations = quanv(block)
                block_tuple = block_to_tuple(block)
                if block_tuple in block_expectation_pairs:
                    print('WARNING: tupelized block already exists in processed data:\n', block_tuple, sep='')
                block_expectation_pairs[block_tuple] = expectations

                if n % 1000 == 0:
                    write_processed_data(block_expectation_pairs)
                    block_expectation_pairs = {}
    except:
        print(f'Interrupted at {n}/{9000 * 24 * 24} blocks.')

    if block_expectation_pairs:
        write_processed_data(block_expectation_pairs)


def block_to_tuple(block: torch.Tensor):
    return tuple(map(int, 1000 * block.flatten()))


def write_processed_data(block_expectation_pairs, data_dir='./processed_data'):
    '''Write the processed data to a pickle file.
    Keys are produced by flattening tensors, multiplying by 1000, taking the floor, and converting to a tuple.
    '''

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    file_nums = [int(f[:-4]) for f in os.listdir(data_dir) if f[:-4].isnumeric()]
    n = max(file_nums) + 1 if file_nums else 0

    with open(os.path.join(data_dir, f'{n}.pkl'), 'wb') as f:
        pickle.dump(block_expectation_pairs, f)


def read_processed_data(data_dir='./processed_data'):
    '''Read the processed data from the pickle file(s)'''

    block_expectation_pairs = {}
    for file in os.listdir(data_dir):
        if file[:-4].isnumeric():
            with open(os.path.join(data_dir, file), 'rb') as f:
                block_expectation_pairs |= pickle.load(f)
    return block_expectation_pairs


def define_balltree_from_processed_data(block_expectation_pairs):
    '''Define the balltree data structure from the processed data'''

    blocks_numpy = np.array(list(block_expectation_pairs.keys()))
    return BallTree(blocks_numpy)


def load_data(ntrain=9000, ntest=1000):
    '''Load the datafile'''
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


def apply_quanv(t, balltree, block_expectation_pairs, kernel_size, nfilters):
    '''Apply the quanvolution operation to a batch of input tensors `t`'''

    t = torch.mean(t, dim=1)

    bs = t.shape[0]
    iout = t.shape[1] - kernel_size + 1
    jout = t.shape[2] - kernel_size + 1
    # Output tensor has shape (bs, nfilters, iout, jout)

    out = torch.empty([bs, nfilters, iout, jout])
    for batch_index in range(bs):
        for i in range(iout):
            for j in range(jout):
                # Get the block as a flattened numpy array
                block = t[batch_index, i : i + kernel_size, j : j + kernel_size]
                block = np.array(block_to_tuple(block)).reshape(1, kernel_size**2)

                # Query the balltree to ge the nearest neighbour quanvolution output
                index = balltree.query(block, return_distance=False)[0][0]
                closest_processed_block = balltree.get_arrays()[0][index]

                # Get the expectation value corresponding to the nearest neighbour
                expectation_values = block_expectation_pairs[tuple(closest_processed_block)]
                out[batch_index, :, i, j] = torch.Tensor(expectation_values)

    return out


# def apply_quanv_parallelized(t, balltree, block_expectation_pairs, kernel_size, nfilters, processes=4):
#     '''Use parallelization to speed up the quanvolution operation'''

#     apply_quanv_partial = partial(
#         apply_quanv,
#         balltree=balltree,
#         block_expectation_pairs=block_expectation_pairs,
#         kernel_size=kernel_size,
#         nfilters=nfilters,
#     )

#     with Pool(processes) as pool:
#         processed_tensors = pool.map(apply_quanv_partial, torch.tensor_split(t, processes))

#     return torch.cat(processed_tensors)


def normalize_quanvolution_output(t, mn, mx):
    t -= mn
    t /= mx - mn
    return t


def train(qnn, dataloader, loss_func, optimizer, balltree, block_expectation_pairs):
    train_loss, train_accuracy = [], []
    softmax = nn.Softmax(dim=1)

    outputs = tuple(chain(*block_expectation_pairs.values()))
    mn, mx = min(outputs), max(outputs)

    qnn.train()
    for x, y in dataloader:
        x = apply_quanv(x, balltree, block_expectation_pairs, 5, 5)
        x = normalize_quanvolution_output(x, mn, mx)

        # Zero gradients and compute the prediction
        optimizer.zero_grad()
        prediction = qnn(x.to(DEVICE))

        # Loss computation and backpropagation
        loss = loss_func(prediction, y)
        loss.backward()

        # Weight optimization
        optimizer.step()

        # Track loss and acuracy metrics
        train_loss.append(loss.item())
        train_accuracy.append(
            (torch.argmax(y, dim=1) == torch.argmax(softmax(prediction), dim=1)).sum().item() / len(y)
        )

    return train_loss, train_accuracy


@torch.no_grad()
def test(qnn, dataloader, loss_func, balltree, block_expectation_pairs):
    test_loss, test_accuracy = [], []
    softmax = nn.Softmax(dim=1)

    outputs = tuple(chain(*block_expectation_pairs.values()))
    mn, mx = min(outputs), max(outputs)

    qnn.eval()
    for x, y in dataloader:
        x = apply_quanv(x, balltree, block_expectation_pairs, 5, 5)
        x = normalize_quanvolution_output(x, mn, mx)

        # Obtain predictions and track loss and accuracy metrics
        prediction = qnn(x.to(DEVICE))
        test_loss.append(loss_func(prediction, y).item())
        test_accuracy.append(
            (torch.argmax(y, dim=1) == torch.argmax(softmax(prediction), dim=1)).sum().item() / len(y)
        )

    return test_loss, test_accuracy


def main(plot=True):
    print("Loading data...")
    # Load the DeepSat-4 dataset
    train_loader, test_loader = load_data(900, 100)

    # Instantiate the model
    qnn = QNN()
    qnn.to(DEVICE)

    # Define the optimizer and loss function
    optimizer = Adam(qnn.parameters(), lr=LR)
    categorical_cross_entropy = nn.CrossEntropyLoss()

    # Get the preprocessed data and define the balltree data structure
    block_expectation_pairs = read_processed_data()
    balltree = define_balltree_from_processed_data(block_expectation_pairs)

    # Training loop
    try:
        print("Training QNN model...")
        train_loss, test_loss = [], []
        train_acc, test_acc = [], []
        for i in range(EPOCHS):
            loss, acc = train(
                qnn,
                train_loader,
                categorical_cross_entropy,
                optimizer,
                balltree,
                block_expectation_pairs,
            )
            train_loss.append(stats.mean(loss))
            train_acc.append(stats.mean(acc))

            loss, acc = test(
                qnn,
                test_loader,
                categorical_cross_entropy,
                balltree,
                block_expectation_pairs,
            )
            test_loss.append(stats.mean(loss))
            test_acc.append(stats.mean(acc))
            print(
                f'Epoch {i + 1}/{EPOCHS}  |  train loss {train_loss[-1]:.4f}  |  train acc {train_acc[-1]:.2%}  |  test loss {test_loss[-1]:.4f}  |  test acc {test_acc[-1]:.2%}'
            )
    except KeyboardInterrupt as e:
        if not test_acc:
            raise KeyboardInterrupt(e)

    if plot:
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

    print(train_acc, test_acc, train_loss, test_loss, sep='\n')
    return train_acc, test_acc, train_loss, test_loss


def run_many(n=4):
    mean_results = np.zeros((4, EPOCHS))
    for i in range(n):
        print(f'Beginning run {i+1}/{n}')
        mean_results += np.array(main(plot=False))
    mean_results /= n
    print('Mean results:')
    for result in mean_results:
        print(list(result))


if __name__ == '__main__':
    # main()
    # run_many(4)
    prerun_quanvolution(start=59, max_cores=2)
