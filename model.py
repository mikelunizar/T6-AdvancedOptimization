import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class MNISTCNN(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, hidden=32, kernel_size=3):
        super().__init__()
        self.save_hyperparameters()

        # CNN layers
        self.conv1 = nn.Conv2d(1, hidden, kernel_size=kernel_size, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, hidden*2, kernel_size=kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden*2)
        self.conv3 = nn.Conv2d(hidden*2, hidden*4, kernel_size=kernel_size, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden*4)

        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden*4 * 3 * 3, hidden*8)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden*8, 10)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # Convolutional layers with ReLU and batch norm
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        # Flatten
        x = x.view(-1, 128 * 3 * 3)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=3, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }