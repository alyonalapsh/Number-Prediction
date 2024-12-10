import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, random_split

import torchvision
from torchvision.transforms import v2

import os
from os import path
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from PIL import Image

import struct
from array import array

from class_MyModel import MyModel


plt.style.use('dark_background')

MNIST_PATH = '../mnist_data'

train_mnist = torchvision.datasets.MNIST(root=MNIST_PATH, train=True, download=True)
test_minst = torchvision.datasets.MNIST(root=MNIST_PATH, train=False, download=True)


def read(dataset):
    if dataset == 'training':
        path_img = MNIST_PATH + 'MNIST/raw/train-images-idx3-ubyte'
        path_lbl = MNIST_PATH + 'MNIST/raw/train-labels-idx1-ubyte'
    elif dataset == "testing":
        path_img = MNIST_PATH + 'MNIST/raw/t10k-images-idx3-ubyte'
        path_lbl = MNIST_PATH + 'MNIST/raw/t10k-labels-idx1-ubyte'
    else:
        raise Exception("unknown dataset")

    with open(path_lbl, 'rb') as f_label:
        _, size = struct.unpack(">II", f_label.read(8))
        lbl = array("b", f_label.read())

    with open(path_img, 'rb') as f_img:
        _, size, rows, cols = struct.unpack(">IIII", f_img.read(16))
        img = array("B", f_img.read())

    return lbl, img, size, rows, cols


def write_dataset(labels, data, size, rows, cols, output_dir):
    classes = {i: f"class_{i}" for i in range(10)}

    output_dirs = [
        path.join(output_dir, classes[i]) for i in range(10)
    ]
    for dir in output_dirs:
        if not path.exists(dir):
            os.makedirs(dir)

    for (i, label) in enumerate(labels):
        output_filename = path.join(output_dirs[label], str(i) + ".jpg")
        # printing is too slow
        # print("writing " + output_filename)

        with open(output_filename, "wb") as h:
            data_i = [
                data[(i * rows * cols + j * cols): (i * rows * cols + (j + 1) * cols)]
                for j in range(rows)
            ]
            data_array = np.asarray(data_i)
            im = Image.fromarray(data_array)
            im.save(output_filename)


output_path = '../mnist/'

for dataset in ["training", "testing"]:
    write_dataset(*read(dataset), path.join(output_path, dataset))


# transfers calculations to the graphics processor if it is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MNISTDataset(Dataset):
    """
    Class for read and save an image.

    Attributes:
        path (str): path to images.
        transform (class): by default=None, transforms the data, normalizes it.
    """

    def __init__(self, path, transform=None):
        """
        The constructor for MNISTDataset class.

        Parameters:
            path (str): path to images.
            transform (class): by default=None, transforms the data, normalizes it.
        """
        self.path = path
        self.transform = transform

        self.len_dataset = 0
        self.data_list = []  # list that contains tuples - path of file and label

        for path_dir, dir_list, file_list in os.walk(path):
            if path_dir == path:
                self.classes = sorted(dir_list)
                self.class_to_idx = {
                    cls_name: i for i, cls_name in enumerate(self.classes)
                }
                continue

            cls = path_dir.split('/')[-1]

            for name_file in file_list:
                file_path = os.path.join(path_dir, name_file)
                self.data_list.append((file_path, self.class_to_idx[cls]))

            self.len_dataset += len(file_list)

    def __len__(self):
        """
        The function that returns length of dataset.

        Returns:
            int: length of dataset.
        """
        return self.len_dataset

    def __getitem__(self, index):
        """
        The function that returns data and their label by index.

        Parameters:
            index (int): index of data in dataset.

        Returns:
            tuple: data and their label.
        """
        file_path, target = self.data_list[index]
        sample = Image.open(file_path)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target


transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.5, ), std=(0.5, ))
    ]
)

train_data = MNISTDataset('../mnist/training', transform=transform)
test_data = MNISTDataset('../mnist/testing', transform=transform)

train_data, val_data = random_split(train_data, [0.7, 0.3])

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

model = MyModel(784, 10).to(device)

loss_model = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), 0.001)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,
                                                          mode='min',
                                                          factor=0.01,
                                                          patience=5
                                                          )

train_loss = []
train_acc = []
val_loss = []
val_acc = []


def model_train(EPOCHS=10):
    """
    Training and validation model.

    Parameters:
        EPOCHS (int): number of training cycles.
    """
    best_loss = None
    count = 0

    for epoch in range(EPOCHS):
        # model training
        model.train()
        running_train_loss = []
        true_answer = 0
        train_loop = tqdm(train_loader, leave=False)

        for x, targets in train_loop:
            x = x.reshape(-1, 28 * 28).to(device)
            targets = targets.reshape(-1).to(torch.int32)
            targets = torch.eye(10)[targets].to(device)

            # forward pass and model loss calculation
            pred = model(x)
            loss = loss_model(pred, targets)

            # backward pass and optimization
            opt.zero_grad()
            loss.backward()
            opt.step()

            running_train_loss.append(loss.item())
            mean_train_loss = sum(running_train_loss) / len(running_train_loss)

            true_answer += (pred.argmax(dim=1) == targets.argmax(dim=1)).sum().item()

            train_loop.set_description(f'Epoch [{epoch + 1}/{EPOCHS}], train_loss = {mean_train_loss: .4f}')

        running_train_acc = true_answer / len(train_data)

        train_loss.append(mean_train_loss)
        train_acc.append(running_train_acc)

        # model validation
        model.eval()
        with torch.no_grad():
            running_val_loss = []
            true_answer = 0

            for x, targets in val_loader:
                x = x.reshape(-1, 28 * 28).to(device)
                targets = targets.reshape(-1).to(torch.int32)
                targets = torch.eye(10)[targets].to(device)

                # forward pass and model loss calculation
                pred = model(x)
                loss = loss_model(pred, targets)

                running_val_loss.append(loss.item())
                mean_val_loss = sum(running_val_loss) / len(running_val_loss)

                true_answer += (pred.argmax(dim=1) == targets.argmax(dim=1)).sum().item()

            running_val_acc = true_answer / len(val_data)

            val_loss.append(mean_val_loss)
            val_acc.append(running_val_acc)

            lr_scheduler.step(mean_val_loss)

            print(
                f'Epoch {epoch + 1}/{EPOCHS}, train loss = {mean_train_loss: .4f}, val loss = {mean_val_loss: .4f}, train accuracy = {running_train_acc: .4f}, val accuracy: {running_val_acc: .4f}')

            if best_loss is None:
                best_loss = mean_val_loss

            if mean_val_loss < best_loss:
                best_loss = mean_val_loss
                count = 0

            if count >= 10:
                print(f'Train stopped at {epoch + 1} epoch')
                break

            count += 1


def model_test():
    """
    Testing model.
    """
    model.eval()

    with torch.no_grad():
        running_test_loss = []

        for x, targets in test_loader:
            x = x.reshape(-1, 28 * 28).to(device)
            targets = targets.reshape(-1).to(torch.int32)
            targets = torch.eye(10)[targets].to(device)

            pred = model(x)
            loss = loss_model(pred, targets)

            running_test_loss.append(loss.item())
            mean_test_loss = sum(running_test_loss) / len(running_test_loss)

        print(f'test loss = {mean_test_loss: .4f}')


def graph():
    """
    Graphs of loss functions and accuracy on training and validation data.
    """
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.legend(['loss_train', 'loss_val'])
    plt.show()

    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.legend(['accuracy_train', 'accuracy_val'])
    plt.show()


def get_pred(idx):
    """
    Outputs the image and model prediction.
    """
    img = test_data[idx]
    model.eval()
    pred = model(img[0].view(1, 28*28).to(device)).detach()
    plt.imshow(img[0].view(28, 28, 1), cmap='gray')
    print(f'Your number is: {np.argmax(pred).item()}')


def save_model(path):
    """
    Saving the model state and parameters.
    """
    checkpoint = {
        'state_model': model.state_dict(),
        'state_opt': opt.state_dict(),
        'state_lr_scheduler': lr_scheduler.state_dict(),
        'loss': {
            'train_loss': train_loss,
            'val_loss': val_loss
        },
        'metric': {
            'train_acc': train_acc,
            'val_acc': val_acc
        }
    }
    torch.save(checkpoint, f'{path}/model_state_dict.pt')
