#
# TOOL:
# Extracts a dictionary from a given model
#
from __future__ import print_function, division

import torch
import sys
import shutil
import os
from PIL import Image

def run():
    if len(sys.argv) != 3:
        print('Wrong argument count, use: python ' +
              __file__+' <model.pl> <dict.pl>')
        quit()
    else:
        model_path = sys.argv[1]
        dict_path = sys.argv[2]

    model = torch.load(model_path)
    if os.path.isfile(dict_path):
        print("Cannot proceed: Target exists")
        quit()
    torch.save(model.state_dict(), dict_path)


if __name__ == '__main__':
    run()
