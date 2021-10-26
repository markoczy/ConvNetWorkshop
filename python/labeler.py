import sys
import os
import torch
from torchvision import transforms
import sys
import os
import os.path
from PIL import Image
import codecs

# Predicts if an Input Image matches a given label ("pos" category)
def predict(model, input):
    ret = model(input)
    if ret[0][0] > 0.0:
        return False
    elif ret[0][1] > 0.0:
        return True
    else:
        return False

# Checks if an input Image matches any of the models ("pos" category) in a
# given folder, returns an array of matched labels/tags
def run():
    if len(sys.argv) != 3:
        print('Wrong argument count, use: python ' +
              __file__+' <model_root> <image>')
        quit()
    else:
        model_root = sys.argv[1]
        image_file = sys.argv[2]

    # Normalize input image according to the trained model
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = Image.open(image_file).convert('RGB')  # must strip alpha channel
    input = transform(img)
    input = input.unsqueeze(0)
    input = input.to('cuda')

    models = {}
    labels = []
    files = os.listdir(model_root)
    # Load models
    for v in files:
        if v.endswith(".pt"):
            label = v[:-3]
            if label.endswith("_fittest"):
                label = label[:-8]
                models[label] = v
            else:
                if not label in models:
                    models[label] = v
    # Test labels
    for label in models:
        model = torch.load(os.path.join(model_root, models[label]))
        model.eval()
        if predict(model, input):
            # hack for windows
            utf8 = bytes(label, 'utf-8')
            labels.append(codecs.decode(utf8, 'mbcs'))
    for l in labels:
        print(l)

if __name__ == '__main__':
    run()
