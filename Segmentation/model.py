import torch
import pytorch_lightning as pl
import torch.nn as nn
from loss import dice_loss
from utils import iou_pytorch
import torchvision
from torch.nn import functional as F


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class ResNetUNet(pl.LightningModule):
    def __init__(self, n_class):
        super().__init__()

        self.base_model = torchvision.models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        # size=(N, 64, x.H/2, x.W/2)
        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        # size=(N, 64, x.H/4, x.W/4)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        # size=(N, 128, x.H/8, x.W/8)
        self.layer2 = self.base_layers[5]
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        # size=(N, 256, x.H/16, x.W/16)
        self.layer3 = self.base_layers[6]
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        # size=(N, 512, x.H/32, x.W/32)
        self.layer4 = self.base_layers[7]
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2,
                                    mode='bilinear',
                                    align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        x_original = self.conv_original_size0(x)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out


class BirdsSegmentationModel(pl.LightningModule):
    def __init__(self, learning_rate, num_classes):
        super().__init__()
        """ Define computations here. """
        self.num_classes = num_classes
        self.lr = learning_rate
        self.model = ResNetUNet(num_classes)
        # freeze backbone layers

        for layer in self.model.base_layers:
            for param in layer.parameters():
                param.requires_grad = False
        self.bce_weight = 0.9

    def forward(self, x):
        """ Use for inference only (separate from training_step). """
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        """the full training loop"""
        x, y = batch

        y_logit = self(x)
        y = torch.unsqueeze(y[:, :, :, 0], dim=1)
        bce = F.binary_cross_entropy_with_logits(y_logit.float(), y.float())
        pred = torch.sigmoid(y_logit)
        dice = dice_loss(pred, y)

        loss = bce * self.bce_weight + dice * (1 - self.bce_weight) * y.size(0)

        return {'loss': loss}

    def configure_optimizers(self):
        """ Define optimizers and LR schedulers. """
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.lr,
                                     weight_decay=5e-4)
        # optimizer = torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                  mode='min',
                                                                  factor=0.1,
                                                                  patience=1,
                                                                  verbose=True)
        lr_dict = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss"
        }

        return [optimizer], [lr_dict]

    # OPTIONAL
    def validation_step(self, batch, batch_idx):
        """the full validation loop"""
        x, y = batch

        y_logit = self(x)
        y = torch.unsqueeze(y[:, :, :, 0], dim=1)

        bce = F.binary_cross_entropy_with_logits(y_logit.float(), y.float())

        pred = torch.sigmoid(y_logit)
        dice = dice_loss(pred, y)
        loss = bce * self.bce_weight + dice * (1 - self.bce_weight) * y.size(0)
        return {'val_loss': loss, 'logs': {'dice': dice, 'bce': bce},
                'iou': iou_pytorch(torch.squeeze(pred, axis=1),
                                   torch.squeeze(y, axis=1))}

    # OPTIONAL
    def training_epoch_end(self, outputs):
        """log and display average train loss and accuracy across epoch"""
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        print(f"| Train_loss: {avg_loss:.3f}")
        self.log('train_loss',
                 avg_loss, prog_bar=True, on_epoch=True, on_step=False)

    # OPTIONAL
    def validation_epoch_end(self, outputs):
        """log and display average val loss and accuracy"""
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        avg_iou = torch.stack([x['iou'] for x in outputs]).mean()
        avg_dice = torch.stack([x['logs']['dice'] for x in outputs]).mean()
        avg_bce = torch.stack([x['logs']['bce'] for x in outputs]).mean()

        print(f"[Epoch {self.trainer.current_epoch:3}] Val_loss: {avg_loss:.\
            3f}, Val_dice: {avg_dice:.\
            3f}, Val_bce: {avg_bce:.3f}, Val_IoU: {avg_iou:.3f}", end=" ")
        self.log('val_loss',
                 avg_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('iou', avg_iou, prog_bar=True, on_epoch=True, on_step=False)
