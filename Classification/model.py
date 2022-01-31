import torch
import pytorch_lightning as pl
import torchvision.models as models


class ResNetClassifier(pl.LightningModule):
    def __init__(self, num_classes, transfer=True, freeze='most'):
        super().__init__()        

        self.resnet_model =  models.resnet18(pretrained=False)
        
        linear_size = list(self.resnet_model.children())[-1].in_features
        
        self.resnet_model.fc = nn.Linear(linear_size, 256)
        self.relu = nn.LeakyReLU(inplace=True)
        self.last = nn.Linear(256, num_classes)

        for child in list(self.resnet_model.children()):
                for param in child.parameters():
                    param.requires_grad = True

        if freeze == 'last':
            for child in list(self.resnet_model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = False
        elif freeze == 'most':
            for child in list(self.resnet_model.children())[:-4]:
                for param in child.parameters():
                    param.requires_grad = False
        elif freeze != 'full':
            raise NotImplementedError('Wrong freezing parameter')
        
    def forward(self, x):
        out = self.resnet_model(x)
        out = self.relu(out)
        out = self.last(out)
        return F.log_softmax(out, dim=1)


class LightningBirdsClassifier(pl.LightningModule):

    def __init__(self, lr_rate=BASE_LR, freeze='most'):
        super(LightningBirdsClassifier, self).__init__()

        self.model = ResNetClassifier(50, True, freeze)

        self.lr_rate = lr_rate

    def forward(self, x):
        return self.model(x)

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, torch.squeeze(y))

        logs = {'train_loss': loss}

        acc = torch.sum(logits.argmax(dim=1) == torch.squeeze(y)) / y.shape[0]
        self.log('train_acc', acc, on_step=True, prog_bar=True)
        self.log('train_loss', loss, on_step=True, prog_bar=True)

        return {'loss': loss, 'log': logs}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, torch.squeeze(y))
        acc = torch.sum(logits.argmax(dim=1) == torch.squeeze(y)) / y.shape[0]
        self.log('val_loss', loss, on_step=True, on_epoch=False)
        self.log('val_acc', acc, on_step=True, on_epoch=False)

        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, torch.squeeze(y))
        acc = torch.sum(logits.argmax(dim=1) == torch.squeeze(y)) / y.shape[0]

        self.log('test_loss', loss, on_step=True, on_epoch=False)
        self.log('test_acc', acc, on_step=True, on_epoch=False)

        return {'test_loss': loss, 'test_acc': acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}
        print('Accuracy: ', round(float(avg_acc), 3))
        self.log('val_acc', avg_acc, on_epoch=True, on_step=False)
        return {'val_loss': avg_loss, 'val_acc': avg_acc, 'log': tensorboard_logs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss, 'test_acc': avg_acc}
        self.log('test_acc', avg_acc, on_epoch=True, on_step=False)
        return {'test_loss': avg_loss, 'test_acc': avg_acc, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr_rate)
        self.optimizer = optimizer
        steps_per_epoch = 24
        scheduler_dict = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
        return [optimizer], [scheduler]



