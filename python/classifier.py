#
# TOOL:
# Classifies an image dataset based on an already trained model and moves the 
# files to the respective subfolders ("pos" for positive match, "neg" for
# negative match)
#
from __future__ import print_function, division

import torch
from torchvision import transforms
import sys
import shutil
import os
from PIL import Image

# Commandline: python classifier.py <model.pl> <imagepath>
# https://discuss.pytorch.org/t/using-model-pth-pytorch-to-predict-image/72935/2


def moveToDir(src, targetDir):
    try:
        os.makedirs(targetDir)
    except:
        pass
    filename = os.path.basename(src)
    target = os.path.join(targetDir, filename)
    shutil.move(src, target)


def predict(model, transform, file):
    img = Image.open(file).convert('RGB')  # must strip alpha channel
    input = transform(img)
    input = input.unsqueeze(0)
    input = input.to('cuda')
    model.eval()
    ret = model(input)
    if ret[0][0] > 0.0:
        return False
    elif ret[0][1] > 0.0:
        return True
    else:
        print("Warning unspecified prediction for file: " + file)
        return False


def run():
    if len(sys.argv) != 3:
        print('Wrong argument count, use: python ' +
              __file__+' <model.pl> <root>')
        quit()
    else:
        model_path = sys.argv[1]
        root_dir = sys.argv[2]

    model = torch.load(model_path)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    files = os.listdir(root_dir)
    print("Found", len(files), "files")
    for v in files:
        file = os.path.join(root_dir, v)
        if os.path.isfile(file):
            print("Processing file: "+file)
            if predict(model, transform, file):
                moveToDir(file, os.path.join(root_dir, 'pos'))
            else:
                moveToDir(file, os.path.join(root_dir, 'neg'))


if __name__ == '__main__':
    run()
