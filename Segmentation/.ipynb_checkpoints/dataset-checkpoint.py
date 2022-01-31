import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import cv2
import os





class BirdsDataset(Dataset):
    def __init__(self, images_path, annotations_path, transform):
        self.images_path = images_path
        self.annotations_path = annotations_path
        self.transform = transform
    
    def __getitem__(self, index):
        image_file, mask_file = self.images_path[index], self.annotations_path[index]
        
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_file)
        mask = cv2.cvtColor(mask, cv2.IMREAD_GRAYSCALE)
        
        transformed = self.transform(image=image, mask=mask)
        return transformed["image"], (transformed["mask"] > 128).long()
        
        
    
    def __len__(self):
        return len(self.images_path)


class BirdsDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, 
                       train_images_path,
                       val_images_path,
                       test_images_path,
                       train_annotations_path,
                       val_annotations_path,
                       test_annotations_path,
                       train_transform,
                       val_transform,
                       test_transform):
        
        self.batch_size = batch_size
        
        self.train_images_path = train_images_path
        self.val_images_path = val_images_path
        self.test_images_path = test_images_path
        
        self.train_annotations_path = train_annotations_path
        self.val_annotations_path = val_annotations_path
        self.test_annotations_path = test_annotations_path
        
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        
    
    def train_dataloader(self):
        train_dataset = BirdsDataset(self.train_images_path, self.train_annotations_path, self.train_transform)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count())
    
    def val_dataloader(self):
        val_dataset = BirdsDataset(self.val_images_path, self.val_annotations_path, self.val_transform)
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count())
    
    def test_dataloader(self):
        test_dataset = BirdsDataset(self.test_images_path, self.test_annotations_path, self.test_transform)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count())
    