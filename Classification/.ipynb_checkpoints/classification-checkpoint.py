import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import TensorDataset, random_split
import torchvision
from torchvision import datasets, transforms, models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import precision_recall_curve
import torchvision.models as models
from sklearn.preprocessing import label_binarize
import cv2
from typing import Tuple, List, Dict, Optional, Any
from torch import Tensor
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.model_selection import train_test_split
from utils import BATCH_SIZE, MAX_EPOCHS, BASE_LR
from model import LightningBirdsClassifier
from dataset import BirdsClassifierModule



def train_classifier(train_gt, train_images_dir, fast_train=False):
    img_folder = train_images_dir
    dm = BirdsClassifierModule(BATCH_SIZE, train_gt, img_folder)
    model = LightningBirdsClassifier() 
    if fast_train:
        trainer = pl.Trainer(max_epochs=1)
        trainer.fit(model, dm)
    else:
        trainer = pl.Trainer(max_epochs=MAX_EPOCHS)
        trainer.fit(model, dm)

    return model


def classify(model_path, test_images_dir):
    model = LightningBirdsClassifier.load_from_checkpoint(model_path)
    transform = A.Compose(
        [
            A.Resize(height=512, width=512),
            A.Normalize(),
            ToTensorV2(),
        ],
    )
    img_classes = {}
    with torch.no_grad():
        for image_file in os.listdir(test_images_dir):
            image = cv2.imread(os.path.join(test_images_dir, image_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = transform(image=image)["image"]
            prediction = model(image[None, ...])
            prediction = torch.argmax(prediction).item()
            img_classes[image_file] = prediction
    return img_classes
