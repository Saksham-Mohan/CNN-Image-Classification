# Please place any imports here.
# BEGIN IMPORTS

import scipy
from scipy import ndimage
import skimage
from skimage import transform
import numpy as np
import cv2
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets

# END IMPORTS

#########################################################
###              BASELINE MODEL
#########################################################

class AnimalBaselineNet(nn.Module):
    def __init__(self, num_classes=16):
        super(AnimalBaselineNet, self).__init__()
        # TODO: Define layers of model architecture
        # TODO-BLOCK-BEGIN
        
        self.conv1 = nn.Conv2d(3, 6, (3,3), stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(6, 12, (3,3), stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(12, 24, (3,3), stride=2, padding=1)
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(1536, 128)
        self.relu4 = nn.ReLU()
        self.cls = nn.Linear(128, num_classes)
        

        # TODO-BLOCK-END

    def forward(self, x):
        x = x.contiguous().view(-1, 3, 64, 64).float()

        # TODO: Define forward pass
        # TODO-BLOCK-BEGIN
#         batch_size = x.shape[0]
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x) 
        x = self.relu4(x)
        x = self.cls(x)

        # TODO-BLOCK-END
        return x

def model_train(net, inputs, labels, criterion, optimizer):
    """
    Will be used to train baseline and student models.

    Inputs:
        net        network used to train
        inputs     (torch Tensor) batch of input images to be passed
                   through network
        labels     (torch Tensor) ground truth labels for each image
                   in inputs
        criterion  loss function
        optimizer  optimizer for network, used in backward pass

    Returns:
        running_loss    (float) loss from this batch of images
        num_correct     (torch Tensor, size 1) number of inputs
                        in this batch predicted correctly
        total_images    (float or int) total number of images in this batch

    Hint: Don't forget to zero out the gradient of the network before the backward pass. We do this before
    each backward pass as PyTorch accumulates the gradients on subsequent backward passes. This is useful
    in certain applications but not for our network.
    """
    # TODO: Foward pass
    # TODO-BLOCK-BEGIN
    
    outputs = net.forward(inputs)
    labels = labels.squeeze() 
    loss = criterion(outputs, labels)
    running_loss = loss.item()
    total_images = inputs.shape[0]
    pred = torch.max(outputs.data, dim=1)[1]
    num_correct = (pred == labels).sum(dim=0)
#     difference = labels - pred
#     num_correct = (difference == 0).sum(dim=0)
    # TODO-BLOCK-END

    # TODO: Backward pass
    # TODO-BLOCK-BEGIN
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # TODO-BLOCK-END

    return running_loss, num_correct, total_images

#########################################################
###               DATA AUGMENTATION
#########################################################

class Shift(object):
    """
  Shifts input image by random x amount between [-max_shift, max_shift]
    and separate random y amount between [-max_shift, max_shift]. A positive
    shift in the x- and y- direction corresponds to shifting the image right
    and downwards, respectively.

    Inputs:
        max_shift  float; maximum magnitude amount to shift image in x and y directions.
    """
    def __init__(self, max_shift=10):
        self.max_shift = max_shift

    def __call__(self, image):
        """
        Inputs:
            image         3 x H x W image as torch Tensor

        Returns:
            shift_image   3 x H x W image as torch Tensor, shifted by random x
                          and random y amount, each amount between [-max_shift, max_shift].
                          Pixels outside original image boundary set to 0 (black).
        """
        image = image.numpy()
        _, H, W = image.shape
        # TODO: Shift image
        rand_xshift = random.uniform(-self.max_shift, self.max_shift)
        rand_yshift = random.uniform(-self.max_shift, self.max_shift)
        image = ndimage.shift(image, np.array([0, rand_yshift, rand_xshift]), order=1, mode='constant')
        image = np.clip(image, 0, 1)
        # TODO-BLOCK-END

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__

class Contrast(object):
    """
    Randomly adjusts the contrast of an image. Uniformly select a contrast factor from
    [min_contrast, max_contrast]. Setting the contrast to 0 should set the intensity of all pixels to the
    mean intensity of the original image while a contrast of 1 returns the original image.

    Inputs:
        min_contrast    non-negative float; minimum magnitude to set contrast
        max_contrast    non-negative float; maximum magnitude to set contrast

    Returns:
        image        3 x H x W torch Tensor of image, with random contrast
                     adjustment
    """

    def __init__(self, min_contrast=0.3, max_contrast=1.0):
        self.min_contrast = min_contrast
        self.max_contrast = max_contrast

    def __call__(self, image):
        """
        Inputs:
            image         3 x H x W image as torch Tensor

        Returns:
            shift_image   3 x H x W torch Tensor of image, with random contrast
                          adjustment
        """
        image = image.numpy()
        _, H, W = image.shape

        # TODO: Change image contrast
        # TODO-BLOCK-BEGIN
        rand_contrast = random.uniform(self.min_contrast, self.max_contrast)
        reshaped_image = np.reshape(image, (3, H*W))
#         calculation of averages for each channel
        averages = np.mean(reshaped_image,axis=1)  

        first_channel_avg = np.zeros((1,H,W)) + averages[0]
        second_channel_avg = np.zeros((1,H,W)) + averages[1]
        third_channel_avg = np.zeros((1,H,W)) + averages[2]

        first_channel_pixels = np.expand_dims(image[0,:,:],axis=0)
        second_channel_pixels = np.expand_dims(image[1,:,:],axis=0)
        third_channel_pixels = np.expand_dims(image[2,:,:],axis=0)
        
        first_channel_pixels = first_channel_avg + (rand_contrast * (first_channel_pixels - first_channel_avg))
        second_channel_pixels = second_channel_avg + (rand_contrast * (second_channel_pixels - second_channel_avg))
        third_channel_pixels = third_channel_avg + (rand_contrast * (third_channel_pixels - third_channel_avg))

        image = np.concatenate((first_channel_pixels, second_channel_pixels, third_channel_pixels), axis=0)
        image = np.clip(image, 0, 1)

        # TODO-BLOCK-END

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__

class Rotate(object):
    """
    Rotates input image by random angle within [-max_angle, max_angle]. Positive angle corresponds to
    counter-clockwise rotation

    Inputs:
        max_angle  maximum magnitude of angle rotation, in degrees


    """
    def __init__(self, max_angle=10):
        self.max_angle = max_angle

    def __call__(self, image):
        """
        Inputs:
            image           image as torch Tensor

        Returns:
            rotated_image   image as torch Tensor; rotated by random angle
                            between [-max_angle, max_angle].
                            Pixels outside original image boundary set to 0 (black).
        """
        image = image.numpy()
        _, H, W  = image.shape

        # TODO: Rotate image
        # TODO-BLOCK-BEGIN

        rand_angle = random.uniform(-self.max_angle, self.max_angle)
        image = scipy.ndimage.rotate(image, rand_angle, axes=(1,2), order=1, reshape = False)
        image = np.clip(image, 0, 1)
        # TODO-BLOCK-END

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__

class HorizontalFlip(object):
    """
    Randomly flips image horizontally.

    Inputs:
        p          float in range [0,1]; probability that image should
                   be randomly rotated
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        """
        Inputs:
            image           image as torch Tensor

        Returns:
            flipped_image   image as torch Tensor flipped horizontally with
                            probability p, original image otherwise.
        """
        image = image.numpy()
        _, H, W = image.shape

        # TODO: Flip image
        # TODO-BLOCK-BEGIN
        rand_num = random.randrange(0,1)
        if rand_num < self.p: #TODO: check whether using the probability like this is correct
            image = cv2.flip(image, 2)
        image = np.clip(image, 0, 1)
        # TODO-BLOCK-END

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__

#########################################################
###             STUDENT MODEL
#########################################################

def get_student_settings(net):
    """
    Return transform, batch size, epochs, criterion and
    optimizer to be used for training.
    """
    dataset_means = [123./255., 116./255.,  97./255.]
    dataset_stds  = [ 54./255.,  53./255.,  52./255.]

    # TODO: Create data transform pipeline for your model
    # transforms.ToPILImage() must be first, followed by transforms.ToTensor()
    # TODO-BLOCK-BEGIN
    
    transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            Contrast(min_contrast=0.3, max_contrast=0.9),
#             Shift(max_shift=5),
            Rotate(max_angle=10),
            HorizontalFlip(p=0.5),
            transforms.Normalize(dataset_means,dataset_stds)
        ])

    # TODO-BLOCK-END

    # TODO: Settings for dataloader and training. These settings
    # will be useful for training your model.
    # TODO-BLOCK-BEGIN
    batch_size = 32

    # TODO-BLOCK-END

    # TODO: epochs, criterion and optimizer
    # TODO-BLOCK-BEGIN

    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(net.parameters(), lr = 0.01)
    epochs = 30
    # TODO-BLOCK-END

    return transform, batch_size, epochs, criterion, optimizer

class AnimalStudentNet(nn.Module):
    def __init__(self, num_classes=16):
        super(AnimalStudentNet, self).__init__()
        # TODO: Define layers of model architecture
        # TODO-BLOCK-BEGIN
        self.conv1 = nn.Conv2d(3, 6, (3,3), stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(6)
        
        self.conv2 = nn.Conv2d(6, 12, (3,3), stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.norm2 = nn.BatchNorm2d(12)
        
        self.conv3 = nn.Conv2d(12, 24, (3,3), stride=2, padding=1)
        self.relu3 = nn.ReLU()
        self.norm3 = nn.BatchNorm2d(24)
    
        self.conv4 = nn.Conv2d(24, 48, (3,3), stride=2, padding=1)
        self.relu4 = nn.ReLU()
        self.norm4 = nn.BatchNorm2d(48)
    
        self.conv5 = nn.Conv2d(48, 96, (3,3), stride=1, padding=1)
        self.relu5 = nn.ReLU()
        self.norm5 = nn.BatchNorm2d(96) 
    
        self.conv6 = nn.Conv2d(96, 192, (3,3), stride=1, padding=1)
        self.relu6 = nn.ReLU()
        self.norm6 = nn.BatchNorm2d(192) 
  
        self.avgpool = nn.AvgPool2d(4)
        
        self.fc = nn.Linear(192, 64)
        self.relu6 = nn.ReLU()
        self.cls = nn.Linear(64, num_classes)
 

        # TODO-BLOCK-END

    def forward(self, x):
        x = x.contiguous().view(-1, 3, 64, 64).float()

        # TODO: Define forward pass
        # TODO-BLOCK-BEGIN
        
#         batch_size = x.shape[0]
        x = self.conv1(x)
        x = self.relu1(x) 
        x = self.norm1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.norm2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.norm3(x)
        
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.norm4(x)
        
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.norm5(x)
        
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.norm6(x) 
        
        x = self.avgpool(x) 
        x = x.view(x.shape[0], -1)
        x = self.fc(x) 
        x = self.relu6(x)    
        x = self.cls(x)

        # TODO-BLOCK-END
        return x

#########################################################
###             ADVERSARIAL IMAGES
#########################################################

def get_adversarial(img, output, label, net, criterion, epsilon):
    """
    Generates adversarial image by adding a small epsilon
    to each pixel, following the sign of the gradient.

    Inputs:
        img        (torch Tensor) image propagated through network
        output     (torch Tensor) output from forward pass of image
                   through network
        label      (torch Tensor) true label of img
        net        image classification model
        criterion  loss function to be used
        epsilon    (float) perturbation value for each pixel

    Outputs:
        perturbed_img   (torch Tensor, same dimensions as img)
                        adversarial image, clamped such that all values
                        are between [0,1]
                        (Clamp: all values < 0 set to 0, all > 1 set to 1)
        noise           (torch Tensor, same dimensions as img)
                        matrix of noise that was added element-wise to image
                        (i.e. difference between adversarial and original image)

    Hint: After the backward pass, the gradient for a parameter p of the network can be accessed using p.grad
    """

    # TODO: Define forward pass
    # TODO-BLOCK-BEGIN
    running_loss = criterion (output, label)
    running_loss.backward()
    img_gradient = img.grad
    perturbed_image = img + (np.sign(img_gradient)*epsilon)
    noise = (np.sign(img_gradient)*epsilon)
    perturbed_image = torch.clamp(perturbed_image, 0, 1)


    # TODO-BLOCK-END

    return perturbed_image, noise

