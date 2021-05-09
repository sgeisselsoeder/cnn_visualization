import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# defining the transformations for the data
transform_imagenet = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.ToTensor(),
    # normalize the images with imagenet data mean and std
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


# Visualise image
def imshow(img, title, batch_size):
    """Custom function to display the image using matplotlib"""

    # define std correction to be made
    std_correction = np.asarray([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    # define mean correction to be made
    mean_correction = np.asarray([0.485, 0.456, 0.406]).reshape(3, 1, 1)

    # convert the tensor img to numpy img and de normalize
    npimg = np.multiply(img.numpy(), std_correction) + mean_correction

    # plot the numpy image
    plt.figure(figsize=(batch_size * 4, 4))
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.savefig("input_image.png")
    plt.show()


# custom function to fetch images from dataloader
def show_batch_images(dataloader, model, classes, batch_size):
    images, _ = next(iter(dataloader))

    # run the model on the images
    outputs = model(images)

    # get the maximum class
    _, pred = torch.max(outputs.data, 1)

    # make grid
    img = torchvision.utils.make_grid(images)

    # call the function
    imshow(img, title=[classes[x.item()] for x in pred], batch_size=batch_size)

    return images, pred
