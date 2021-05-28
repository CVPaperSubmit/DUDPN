
import time
import torch
from torch.backends import cudnn
from matplotlib import colors

from backbone import EfficientDetBackbone
import cv2
import numpy as np
from torchstat import stat


model= EfficientDetBackbone(1,2)
stat(model,(3,768,768))