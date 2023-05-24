import pickle
import os
import shelve
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
from pympler import asizeof

from quanvolution_v1 import Quanvolution
from constants import FILTERS

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

DATAFILE = './deepsat4/sat-4-full.mat'  # https://csc.lsu.edu/~saikat/deepsat/
BATCH_SIZE = 128
LR = 0.001
EPOCHS = 100


class Data(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.Tensor(x_data).permute((2, 0, 1, 3))

        # # Normalize the input data
        # self.x_data -= self.x_data.mean(dim=(0, 1, 2))
        # self.x_data /= self.x_data.std(dim=(0, 1, 2))

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
    def __init__(self):
        super().__init__()

        # self.pool1 = nn.AvgPool2d(kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=12, stride=1, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=1452, out_features=4)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x = self.relu(self.pool1(x))  # Input already has the quanvolutional layer applied
        x = self.relu(x)
        x = self.relu(self.pool2(self.conv2(x)))
        x = self.fc(self.flatten(x))
        return x


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
                if n < 109313:  #######################################################################################
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
    file_nums = [int(f[:-4]) for f in os.listdir(dir) if f[:-4].isnumeric() or f[0] == '-']
    n = min(file_nums) if file_nums else 0  ###########################################################################
    file = os.path.join(dir, f'{n-1}.pkl')  ###########################################################################
    print(f'Writing \'{file}\'')
    with open(file, 'wb') as f:
        pickle.dump(block_expectation_pairs, f)


def define_balltree_from_pickle_files(directory='processed_blocks'):
    '''Load the pickle files containing the tensor blocks and their values after quanvolution'''
    all_blocks = {}
    for file in os.listdir(directory):
        if not (file[:-4].isnumeric() or file[0] == '-'):
            continue
        with open(os.path.join(directory, file), 'rb') as f:
            block_expectation_pairs = pickle.load(f)

        all_blocks |= block_expectation_pairs

    blocks_flattened_numpy = np.array([x.flatten().numpy() for x in list(all_blocks.keys())])
    all_blocks = {tuple(x.flatten().numpy()): y for x, y in all_blocks.items()}
    balltree = BallTree(blocks_flattened_numpy)

    print(f'BallTree size: {asizeof.asizeof(balltree) / (1024**2):.6f} MB')
    print(f'Dictionary size: {asizeof.asizeof(all_blocks) / (1024**2):.6f} MB')
    print(f'Processed blocks: {len(all_blocks)}')

    return all_blocks, balltree


def tuple_to_str(key):
    return ','.join(map(str, key))


def str_to_tuple(s):
    return tuple(map(float, s.split(',')))


def define_balltree_from_pickle_files_shelve(directory='processed_blocks', output_file='merged_blocks.shelve'):
    '''
    Load the pickle files containing the tensor blocks and their values after quanvolution.
    Here we use the shelve module to avoid excessive memory use when the files are large.
    '''
    if not os.path.exists(output_file + '.db'):
        with shelve.open(output_file, 'n') as merged_blocks:
            for file in os.listdir(directory):
                if not (file[:-4].isnumeric() or file[0] == '-'):
                    continue
                with open(os.path.join(directory, file), 'rb') as f:
                    block_expectation_pairs = pickle.load(f)

                for key, value in block_expectation_pairs.items():
                    key = tuple_to_str(tuple(key.flatten().numpy()))
                    if key not in merged_blocks:
                        merged_blocks[key] = value
                    else:
                        print(f'Dupicate key: {key}')

                print(f'Merged {file}')
            print('Merging completed')
            all_blocks = {str_to_tuple(x): y for x, y in merged_blocks.items()}
    else:
        print('Merged file exists. Reading values...')
        with shelve.open(output_file, 'r') as merged_blocks:
            all_blocks = {str_to_tuple(x): y for x, y in merged_blocks.items()}

    blocks_flattened_numpy = np.array(list(all_blocks.keys()))
    balltree = BallTree(blocks_flattened_numpy)

    print(f'    BallTree size: {asizeof.asizeof(balltree) / (1024**2):.6f} MB')
    print(f'    Dictionary size: {asizeof.asizeof(all_blocks) / (1024**2):.6f} MB')
    print(f'    Processed blocks: {len(all_blocks)}')

    return all_blocks, balltree


def get_nearest_neighbour_quanvolution_outputs(
    dataloader, balltree, block_expectation_pairs, kernel_size, nfilters, parallel=True, processes=4, normalize=True
):
    all_x_tensors = torch.cat([t for t, _ in dataloader])

    if parallel:
        quanvoluted = apply_quanv_parallelized(
            all_x_tensors, balltree, block_expectation_pairs, kernel_size, nfilters, processes
        )
    else:
        quanvoluted = apply_quanv(all_x_tensors, balltree, block_expectation_pairs, kernel_size, nfilters)

    if normalize:
        outputs = tuple(chain(*block_expectation_pairs.values()))
        quanvoluted = normalize_quanvolution_output(quanvoluted, min(outputs), max(outputs))

    return {
        tuple(int(100 * v) for v in tuple(t.flatten().numpy())): quanvoluted[i] for i, t in enumerate(all_x_tensors)
    }


def get_batch_quanvolution_output_from_precomputed(batch, quanv_input_output_pairs):
    if isinstance(batch, torch.Tensor):
        batch = [tuple(int(100 * v) for v in tuple(t.flatten().numpy())) for t in batch]
    x = [quanv_input_output_pairs.get(t) for t in batch]
    if any(t is None for t in x):
        print(f'{sum([t is None for t in x])} / {len(x)} are None!!')
        return None
    return torch.stack(x)


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


def apply_quanv(t, balltree, block_expectation_pairs, kernel_size, nfilters):
    t = torch.mean(t, dim=1)

    bs = t.shape[0]
    iout = t.shape[1] - kernel_size + 1
    jout = t.shape[2] - kernel_size + 1
    # Output tensor has shape (bs, nfilters, iout, jout)

    out = torch.empty([bs, nfilters, iout, jout])
    for batch_index in range(bs):
        for i in range(iout):
            for j in range(jout):
                block = t[batch_index, i : i + kernel_size, j : j + kernel_size].reshape(1, kernel_size**2).numpy()
                index = balltree.query(block, return_distance=False)[0][0]
                closest_processed_block = tuple(balltree.get_arrays()[0][index])
                expectation_values = block_expectation_pairs[closest_processed_block]
                out[batch_index, :, i, j] = torch.Tensor(expectation_values)

    return out


def apply_quanv_parallelized(t, balltree, block_expectation_pairs, kernel_size, nfilters, processes=4):
    apply_quanv_partial = partial(
        apply_quanv,
        balltree=balltree,
        block_expectation_pairs=block_expectation_pairs,
        kernel_size=kernel_size,
        nfilters=nfilters,
    )

    with Pool(processes) as pool:
        processed_tensors = pool.map(apply_quanv_partial, torch.tensor_split(t, processes))

    return torch.cat(processed_tensors)


def normalize_quanvolution_output(t, mn, mx):
    t -= mn
    t /= mx - mn
    return t


def train(qnn, dataloader, loss_func, optimizer, balltree, block_expectation_pairs, quanv_input_output_train_pairs):
    train_loss, train_accuracy = [], []
    softmax = nn.Softmax(dim=1)

    outputs = tuple(chain(*block_expectation_pairs.values()))
    mn, mx = min(outputs), max(outputs)

    qnn.train()
    for x, y in dataloader:
        if (x_quanvolved := get_batch_quanvolution_output_from_precomputed(x, quanv_input_output_train_pairs)) is None:
            print('WARNING: Training tensor quanvolution not precomputed. Computing quanvolution manually.')
            x = apply_quanv_parallelized(x, balltree, block_expectation_pairs, 5, 5)
            x = normalize_quanvolution_output(x, mn, mx)
        else:
            x = x_quanvolved

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
def test(qnn, dataloader, loss_func, balltree, block_expectation_pairs, quanv_input_output_test_pairs):
    test_loss, test_accuracy = [], []
    softmax = nn.Softmax(dim=1)

    outputs = tuple(chain(*block_expectation_pairs.values()))
    mn, mx = min(outputs), max(outputs)

    qnn.eval()
    for x, y in dataloader:
        if (x_quanvolved := get_batch_quanvolution_output_from_precomputed(x, quanv_input_output_test_pairs)) is None:
            print('WARNING: Testing tensor quanvolution not precomputed. Computing quanvolution manually.')
            x = apply_quanv_parallelized(x, balltree, block_expectation_pairs, 5, 5)
            x = normalize_quanvolution_output(x, mn, mx)
        else:
            x = x_quanvolved

        # Obtain predictions and track loss and accuracy metrics
        prediction = qnn(x.to(DEVICE))
        test_loss.append(loss_func(prediction, y).item())
        test_accuracy.append(
            (torch.argmax(y, dim=1) == torch.argmax(softmax(prediction), dim=1)).sum().item() / len(y)
        )

    return test_loss, test_accuracy


def write_quanv_in_out_dict_data(quanv_pairs, file):
    shape = ','.join(str(n) for n in list(quanv_pairs.values())[0].shape) + '\n'
    lines = [shape]
    for key, value in quanv_pairs.items():
        key_str = ','.join(map(str, key)) + '\n'
        value_str = ','.join(map(str, tuple(value.flatten().numpy()))) + '\n'
        lines.append(key_str)
        lines.append(value_str)
    file.writelines(lines)


def read_quanv_in_out_dict_data(file):
    quanv_pairs = {}
    lines = file.readlines()
    shape = list(map(int, lines[0].split(',')))
    lines = lines[1:]
    for i in range(len(lines) // 2):
        key_line = lines[2 * i]
        value_line = lines[2 * i + 1]
        key = tuple(map(int, key_line.split(',')))
        value = torch.Tensor(list(map(float, value_line.split(',')))).reshape(shape)
        quanv_pairs[key] = value
    return quanv_pairs


def main(epochs=None, lr=None, batch_size=None, plot=True):
    global EPOCHS, LR, BATCH_SIZE
    EPOCHS, LR, BATCH_SIZE = epochs or EPOCHS, lr or LR, batch_size or BATCH_SIZE

    print("Loading data...")
    # Load the DeepSat-4 dataset
    train_loader, test_loader = load_data(9000, 1000)

    print('Building BallTree...')
    # Load the data from the pre-run convolution operation
    block_expectation_pairs, balltree = define_balltree_from_pickle_files_shelve()

    # Precompute the input-output pairs of the quanvolution layer
    print('Precomputing quanvolution output...')
    train_file = 'train_quanv_precomputed.txt'
    if os.path.exists(train_file):
        print('    Train file exists, reading...')
        with open(train_file, 'r') as f:
            quanv_input_output_train_pairs = read_quanv_in_out_dict_data(f)
    else:
        quanv_input_output_train_pairs = get_nearest_neighbour_quanvolution_outputs(
            train_loader, balltree, block_expectation_pairs, 5, 5, processes=6
        )
        with open(train_file, 'w') as f:
            write_quanv_in_out_dict_data(quanv_input_output_train_pairs, f)

    test_file = 'test_quanv_precomputed.txt'
    if os.path.exists(test_file):
        print('    Test file exists, reading...')
        with open(test_file, 'r') as f:
            quanv_input_output_test_pairs = read_quanv_in_out_dict_data(f)
    else:
        quanv_input_output_test_pairs = get_nearest_neighbour_quanvolution_outputs(
            test_loader, balltree, block_expectation_pairs, 5, 5, processes=6
        )
        with open(test_file, 'w') as f:
            write_quanv_in_out_dict_data(quanv_input_output_test_pairs, f)

    # assert all(
    #     tuple(int(100 * v) for v in tuple(t.flatten().numpy())) in quanv_input_output_train_pairs
    #     for t in torch.cat([t for t, _ in train_loader])
    # )
    # assert all(
    #     tuple(int(100 * v) for v in tuple(t.flatten().numpy())) in quanv_input_output_test_pairs
    #     for t in torch.cat([t for t, _ in test_loader])
    # )
    print('Quanvolution output obtained successfully.')

    # Instantiate the model
    qnn = QNN()
    qnn.to(DEVICE)

    # Define the optimizer and loss function
    optimizer = Adam(qnn.parameters(), lr=LR)
    categorical_cross_entropy = nn.CrossEntropyLoss()

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
                quanv_input_output_train_pairs,
            )
            train_loss.append(stats.mean(loss))
            train_acc.append(stats.mean(acc))

            loss, acc = test(
                qnn,
                test_loader,
                categorical_cross_entropy,
                balltree,
                block_expectation_pairs,
                quanv_input_output_test_pairs,
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
    run_many(4)
    # prerun_quanvolution()

nn.Module.eval