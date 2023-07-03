from torch.optim import lr_scheduler
from torchvision import transforms
import functools
import torch.nn as nn

import json
import math
import torch
import random
import numpy as np


def get_scheduler(optimizer, last_epoch, decay_epochs, type="step"):
    if type == "step":
      def lambda_rule(epoch):
        factor = last_epoch + epoch - decay_epochs
        if factor <= 0:
            return 1.0
        return 1.0 - factor / float(decay_epochs + 1)  # +1 so that lr will never truly be 0
      return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    else:
      return lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, threshold=0.01, patience=5)



def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_norm_layer(norm_type='batch'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


class Options:
    def __init__(self):
        f = open("./Scriptsv2/options.json")
        options = json.load(f)
        f.close()

        for key, value in options.items():
            self.__dict__[key] = value

    def __setattr__(self, name, value):
        raise Exception("Options are readonly")


class DecayingBlur:
    def __init__(self, kernel_size=13, max_sigma=40, num_epochs=150):
        self.kernel_size = kernel_size
        self.max_sigma = max_sigma
        self.num_epochs = num_epochs
        self.step = num_epochs // 4

    def __call__(self, img, epoch):
        if epoch >= self.num_epochs:
            return img
        
        decaying_factor = (self.num_epochs - epoch) / (self.num_epochs + 1)  
        sigma = self.max_sigma * decaying_factor
        #print(sigma)
        kernel_size = self.kernel_size - 2 * (epoch // self.step)

        # Apply Gaussian blur
        img = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)(img)
        return img


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((290, 290)),
    transforms.RandomCrop(size=256),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


eval_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

predict_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def seed_everything(seed=13):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


