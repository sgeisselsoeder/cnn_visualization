# Filter visualisation with pretrained model
# derived from https://towardsdatascience.com/visualizing-convolution-neural-networks-using-pytorch-3dfa8443e74e

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def plot_filters_single_channel(t, layer_num):
    # kernels depth * number of kernels
    nplots = t.shape[0]*t.shape[1]
    ncols = 12

    nrows = 1 + nplots//ncols
    # convert tensor to numpy image
    npimg = np.array(t.numpy(), np.float32)

    count = 0
    fig = plt.figure(figsize=(ncols, nrows))

    # looping through all the kernels in each channel
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            count += 1
            ax1 = fig.add_subplot(nrows, ncols, count)
            npimg = np.array(t[i, j].numpy(), np.float32)
            npimg = (npimg - np.mean(npimg)) / np.std(npimg)
            npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
            ax1.imshow(npimg)
            ax1.set_title(str(i) + ',' + str(j))
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])

    plt.tight_layout()
    plt.savefig("filters_single_channel" + str(layer_num) + ".png", dpi=100)
    plt.close()


def plot_filters_multi_channel(t, layer_num):
    # get the number of kernals
    num_kernels = t.shape[0]

    # define number of columns for subplots
    num_cols = 12
    # rows = num of kernels
    num_rows = num_kernels

    # set the figure size
    fig = plt.figure(figsize=(num_cols, num_rows))

    # looping through all the kernels
    for i in range(t.shape[0]):
        ax1 = fig.add_subplot(num_rows, num_cols, i+1)

        # for each kernel, we convert the tensor to numpy
        npimg = np.array(t[i].numpy(), np.float32)
        # standardize the numpy image
        npimg = (npimg - np.mean(npimg)) / np.std(npimg)
        npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
        npimg = npimg.transpose((1, 2, 0))
        ax1.imshow(npimg)
        ax1.axis('off')
        ax1.set_title(str(i))
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.tight_layout()
    plt.savefig("filters_multi_channel" + str(layer_num) + ".png", dpi=100)
    plt.close()


def plot_weights(model, layer_num, single_channel=True):
    # extracting the model features at the particular layer number
    layer = model.features[layer_num]

    # checking whether the layer is convolution layer or not
    if isinstance(layer, torch.nn.Conv2d):
        # getting the weight tensor data
        weight_tensor = model.features[layer_num].weight.data

        if single_channel:
            plot_filters_single_channel(weight_tensor, layer_num=layer_num)

        else:
            if weight_tensor.shape[1] == 3:
                plot_filters_multi_channel(weight_tensor, layer_num=layer_num)
            else:
                print("Can only plot weights with three channels with single channel = False")

    else:
        print("Can only visualize layers which are convolutional")


# We will use alexnet pretrained with imagenet data
alexnet = torchvision.models.alexnet(pretrained=True)
print(alexnet)

# visualize weights - first conv layer
plot_weights(alexnet, 0, single_channel=False)
plot_weights(alexnet, 0, single_channel=True)
