# -*- coding: utf-8 -*-
"""
================================================================================
CNN Transfer learning Breeder
================================================================================

Trains a model of a binary classifier (positive, negative) for a given tag/class

Usage: python breeder.py <dataset>

The Dataset folder structure is expected to be as follows:


data/<dataset>:
  - train:
    - pos
    - neg
  - val:
    - pos
    - neg

Based on the work from Sasank Chilamkurthy at:
https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

(Original work is licensed BSD)
"""

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import sys

from torchvision.transforms.transforms import RandomRotation


# Dataloader Config ########################################################
dataset = "ant"
max_rotation_deg = 20
resize_scale_range = (0.08, 1.0)
batch_size = 4
workers = 1
# Evolution Params #########################################################
generations = 30
epochs = 35
strict_selection = False    # Pure darwinism: Only the fittest survives
force_update = False        # CAUTON: Only use for changed datasets
# Hyperparameters ##########################################################
learn_rate = 0.001
learn_rate_momentum = 0.9
learn_rate_decay_epochs = 7
learn_rate_decay_gamma = 0.2
############################################################################

# Initialises CUDA if machine is CUDA capable otherwise uses CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Sets plotter to interactive mode
def init_plotter():
    plt.ion()

# Plots an image from a tensor
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def createModel():
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)
    return model


def loadModel(name):
    model_ft = torch.load(name)
    model_ft.eval()
    return model_ft

# Trains Model with the given training set of the dataset
# (dataset/train/**) returns the trained model and the best accuracy reached
def train_model(dataloaders, dataset_sizes, model, criterion, optimizer, scheduler, last_acc, num_epochs=25):
    since = time.time()

    # Store the input model as it is currently the best
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = last_acc

    # Range through epochs and process training
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc


# Evaluates accuracy of a given model using the validation set
# of the dataset, returns accuracy on the validation set
def eval(dataloaders, dataset_sizes, model, criterion, optimizer):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in dataloaders['val']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    # epoch_loss = running_loss / dataset_sizes['val']
    epoch_acc = running_corrects.double() / dataset_sizes['val']
    return epoch_acc

# (currently unused) Could be used to visualize a dataset
def visualize_data(dataloaders, class_names, model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


# The main function
def run():
    if len(sys.argv) != 2:
        print('Please specify dataset')
        quit()
    else:
        dataset = sys.argv[1]

    data_dir = '../data/' + dataset
    model_name = '../model/' + dataset+'.pt'
    fittest_name = '../model/' + dataset + '_fittest.pt'

    init_plotter()
    print("Using device:", device)

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=resize_scale_range),
            transforms.RandomRotation(max_rotation_deg),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}

    # Init model (load or create)
    if os.path.isfile(model_name):
        model = loadModel(model_name)
    else:
        model = createModel()
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=learn_rate, momentum=learn_rate_momentum)

    # Init Dataloader for train and validation set
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=workers)
        for x in ['train', 'val']
    }
    dataset_sizes = {
        x: len(image_datasets[x])
        for x in ['train', 'val']
    }

    # Validate current model
    acc = eval(dataloaders, dataset_sizes, model, criterion, optimizer_ft)
    fittest_acc = acc

    # If there is already a fittest model, load it too
    if os.path.isfile(fittest_name):
        fittest = loadModel(fittest_name)
        fittest_acc = eval(dataloaders, dataset_sizes,
                           fittest, criterion, optimizer_ft)
        if strict_selection and fittest_acc > acc:
            model = fittest
            acc = fittest_acc
    # If there is no fittest model, save current as the fittest
    else:
        torch.save(model, fittest_name)

    print("Accuracy of loaded model: {:4f}".format(acc))
    if not strict_selection:
        print("Accuracy of fittest model: {:4f}".format(fittest_acc))

    if force_update:
        print('CAUTION: Force Update is enabled, next breed will be overwritten')
        fittest_acc = 0.0
        acc = 0.0

    for gen in range(generations):
        print("-------------------------------------------------")
        print("-- Generation ", gen)
        print("-------------------------------------------------")
        print("")
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=1) for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

        # Set Learnrate Decay for each training step
        exp_lr_scheduler = lr_scheduler.StepLR(
            optimizer_ft, step_size=learn_rate_decay_epochs, gamma=learn_rate_decay_gamma)

        set_acc = acc if strict_selection else 0.0
        model, acc = train_model(dataloaders, dataset_sizes, model, criterion, optimizer_ft, exp_lr_scheduler, set_acc, num_epochs=epochs)

        # always save the fittest model when not in strict selection mode
        if acc > fittest_acc and not strict_selection:
            fittest_acc = acc
            torch.save(model, fittest_name)
            print("New fittest, saved as "+fittest_name)

        # save fittest of every generation
        torch.save(model, model_name)
        print("Saved current model as "+model_name)


if __name__ == '__main__':
    run()
