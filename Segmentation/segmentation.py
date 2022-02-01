import glob
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import albumentations as A
from albumentations.pytorch import ToTensorV2
from os.path import join

from utils import BATCH_SIZE, LEARNING_RATE, NUM_CLASSES, MAX_EPOCHS
from model import BirdsSegmentationModel
from dataset import BirdsDataModule


def train_model(train_data_path):
    gt_dir = join(train_data_path, 'gt')
    images_dir = join(train_data_path, 'images')

    gt_filenames = np.sort(glob.glob(join(gt_dir, '**/*.png')))
    images_filenames = np.sort(glob.glob(join(images_dir, '**/*.jpg')))

    n = len(images_filenames)

    assert len(gt_filenames) == n
    checker = []
    for i in range(n):
        # for specific problem
        a = images_filenames[i][93:-4]
        b = gt_filenames[i][89:-4]
        checker.append(a == b)
    assert np.sum(checker) == n

    TRAIN_SPLIT = int(n * 0.9)

    train_images_path = images_filenames[:TRAIN_SPLIT]
    val_images_path = images_filenames[TRAIN_SPLIT:]
    test_images_path = val_images_path

    train_annotations_path = gt_filenames[:TRAIN_SPLIT]
    val_annotations_path = gt_filenames[TRAIN_SPLIT:]
    test_annotations_path = val_annotations_path

    train_transform = A.Compose([A.RandomResizedCrop(320, 320),
                                 A.ColorJitter(),
                                 A.Normalize(),
                                 ToTensorV2()])
    val_transform = A.Compose([A.Resize(320, 320),
                               A.Normalize(),
                               ToTensorV2()])
    test_transform = val_transform

    birds_data_module = BirdsDataModule(BATCH_SIZE,
                                        train_images_path,
                                        val_images_path,
                                        test_images_path,
                                        train_annotations_path,
                                        val_annotations_path,
                                        test_annotations_path,
                                        train_transform,
                                        val_transform,
                                        test_transform)
    model = BirdsSegmentationModel(LEARNING_RATE, NUM_CLASSES)
    # Save the model periodically by monitoring a quantity.
    MyModelCheckpoint = ModelCheckpoint(dirpath='runs/pl_segmentation',
                                        filename='{epoch}-{iou:.3f}',
                                        monitor='iou',
                                        mode='max',
                                        save_top_k=3)
    # Monitor a metric and stop training when it stops improving.
    MyEarlyStopping = EarlyStopping(monitor="val_loss",
                                    mode="min",
                                    patience=20,
                                    verbose=True)

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        gpus=-1,
        precision=16,
        callbacks=[MyEarlyStopping, MyModelCheckpoint]
    )

    trainer.fit(model, birds_data_module)


if __name__ == "__main__":
    train_data_path = "../public_tests/00_test_val_input/train"
    train_model(train_data_path)
