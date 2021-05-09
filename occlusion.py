# Occlusion analysis with pretrained model
# derived from https://towardsdatascience.com/visualizing-convolution-neural-networks-using-pytorch-3dfa8443e74e

import torch
import torchvision
import seaborn as sns
import numpy as np
from src.cnn_vis_utils import transform_imagenet, show_batch_images
import warnings
warnings.filterwarnings("ignore")


# reading the labels of data we uploaded
with open("data/imagenet_labels.txt") as f:
    classes = eval(f.read())

# define the data we uploaded as evaluation data and apply the transformations
evalset = torchvision.datasets.ImageFolder(root="data/imagenet", transform=transform_imagenet)

# create a data loader for evaluation
batch_size = 1  # batch size
evalloader = torch.utils.data.DataLoader(evalset, batch_size=batch_size, shuffle=True)

# looking at data using iter
dataiter = iter(evalloader)
images, labels = dataiter.next()
# shape of images bunch
print(images.shape)
# shape of single image in a bunch
print(images[0].shape)

# Load pretrained vgg16 model pretrained on imagenet data
model = torchvision.models.vgg16(pretrained=True)
model.eval()

images, pred = show_batch_images(dataloader=evalloader, model=model, classes=classes, batch_size=batch_size)


# Occlusion analysis
# running inference on the images, first without occlusion

# vgg16 pretrained model
outputs = model(images)
print(outputs.shape)

# passing the outputs through softmax to interpret them as probability
outputs = torch.nn.functional.softmax(outputs, dim=1)

# getting the maximum predicted label
prob_no_occ, pred = torch.max(outputs.data, 1)

# get the first item
prob_no_occ = prob_no_occ[0].item()
print(prob_no_occ)


# conduct occlusion experiments
def occlusion(model, image, label, occ_size=50, occ_stride=50, occ_pixel=0.5):
    # get the width and height of the image
    width, height = image.shape[-2], image.shape[-1]

    # setting the output image width and height
    output_height = int(np.ceil((height-occ_size)/occ_stride))
    output_width = int(np.ceil((width-occ_size)/occ_stride))

    # create a white image of sizes we defined
    heatmap = torch.zeros((output_height, output_width))

    # iterate all the pixels in each column
    for h in range(0, height):
        for w in range(0, width):

            h_start = h*occ_stride
            w_start = w*occ_stride
            h_end = min(height, h_start + occ_size)
            w_end = min(width, w_start + occ_size)

            if (w_end) >= width or (h_end) >= height:
                continue

            input_image = image.clone().detach()

            # replacing all the pixel information in the image with occ_pixel(grey) in the specified location
            input_image[:, :, w_start:w_end, h_start:h_end] = occ_pixel

            # run inference on modified image
            output = model(input_image)
            output = torch.nn.functional.softmax(output, dim=1)
            prob = output.tolist()[0][label]

            # setting the heatmap location to probability value
            heatmap[h, w] = prob

    return heatmap


heatmap = occlusion(model, images, pred[0].item(), 32, 14)

# displaying the image using seaborn heatmap and also setting the maximum value of gradient to probability
imgplot = sns.heatmap(heatmap, xticklabels=False, yticklabels=False, vmax=prob_no_occ)
figure = imgplot.get_figure()
figure.savefig('occlusion_heatmap.png', dpi=400)
