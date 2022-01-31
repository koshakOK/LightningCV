import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import cv2



class BirdsClassifierDataset(Dataset):
    def __init__(self, images_path, targets, method, transform=None):
        self.images_path = images_path
        self.method = method
        if method == 'train':
            self.image_files = np.sort(os.listdir(images_path))[:2500]
        elif method == 'val':
            self.image_files = np.sort(os.listdir(images_path))[2000:2500]
        else:
            self.image_files = np.sort(os.listdir(images_path))[2200:]
        self.targets = [
            targets.loc[image_file].values for image_file in self.image_files
        ]
        self.transform = transform
 
    def __len__(self):
        return len(self.image_files)
 
    def __getitem__(self, index):
        image_file, target = self.image_files[index], self.targets[index]
        image = cv2.imread(os.path.join(self.images_path, image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)['image']
            target = torch.tensor(target)
        return image, target

class BirdsClassifierModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_gt, val_gt, test_gt):
        super().__init__()
        self.batch_size = batch_size
        self.images_path = 'public_tests/00_test_img_input/train/images/kek'
        self.train_targets = train_gt
        self.val_targets = val_gt
        self.test_targets = test_gt
        self.train_transform = A.Compose(
        [A.SmallestMaxSize(max_size=640),
            A.RandomResizedCrop(height=512, width=512),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.CoarseDropout(),
            A.Normalize(),
            ToTensorV2()]
        )
        self.val_transform = A.Compose(
            [A.Resize(512, 512), A.Normalize(), ToTensorV2()]
        )
        self.test_transform = A.Compose(
            [A.Resize(512, 512), A.Normalize(), ToTensorV2()]
        )        
    
    def train_dataloader(self):
        self.train_dataset = BirdsClassifierDataset(self.images_path, self.train_targets, 'train', self.train_transform)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=os.cpu_count(), shuffle=True)

    def val_dataloader(self):
        self.val_dataset = BirdsClassifierDataset(self.images_path, self.val_targets, 'val', self.val_transform)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=os.cpu_count(), shuffle=False)

    def test_dataloader(self):
        self.test_dataset = BirdsClassifierDataset(self.images_path, self.test_targets, 'test', self.test_transform)
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=os.cpu_count(), shuffle=False)

