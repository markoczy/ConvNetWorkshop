# ConvNetWorkshop

CNN Transfer Learning Workshop for the Digital Impact Network.

## Requirements

- Python 3.9.5 https://www.python.org 
- Pytorch mit CUDA 11 Support 1.8.1 https://pytorch.org/
- NVIDIA CUDA Toolkit 11.1 https://developer.nvidia.com/cuda-toolkit
- (Optional) For the Frontend: Go1.15 or later

## Modules

- **python/breeder.py:** Trains a model of a given label using the transfer learning method on a ResNet18.
- **python/labeler.py:** Labels a given Image using the pretrained models inside the provided directory.
- **go/labeler-frontend:** Web Frontend for the labeler.

## Workflow

- Get your data ready:
  - Create one folder for each label you want to train inside the `data` folder, the name corresponds the name of the label (e.g. "airplane")
  - Create a folder named `train` and a folder named `val` inside your label folder
  - Create a folder named `pos` and a folder named `neg` in the `train` and `val` folders
  - Copy your training data inside the corresponding folders the folder structure should look as follows:

```yaml
- data:
  - label1:
    - train:
      - pos:
        img1.png
        img2.png
        ...
      - neg:
        img3.png
        img4.png
        ...
    - val:
      - pos:
        img5.png
        img6.png
        ...
      - neg:
        img7.png
        img8.png
        ...
  - label2:
    ...
```

- Train the labels:
  - Open a console inside the `python` folder
  - (Optional) Adjust the parameters inside the config section of `breeder.py`
  - Call `python breeder.py <labelname>` to start training one label
  - Repeat for each label
- Use the labeler:
  - Go in the folder `go/labeler-frontend`
  - Call `go build` and then `labeler-frontend`
  - Open the browser at `http://localhost:7890` and label a picture by uploading it to the webserver
