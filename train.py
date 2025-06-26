import os
import wandb
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from torchvision import transforms, datasets
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

def load_mnist(fraction=0.25):
    transform = transforms.ToTensor()
    mnist_train_full = datasets.MNIST(os.getcwd(), train=True, download=True, transform=transform)

    # Reduce training set if fraction < 1
    if 0 < fraction < 1.0:
        subset_size = int(fraction * len(mnist_train_full))
        mnist_train_full, _ = random_split(mnist_train_full, [subset_size, len(mnist_train_full) - subset_size])

    # Extract images and labels as tensors
    data = torch.stack([mnist_train_full[i][0] for i in range(len(mnist_train_full))])
    targets = torch.tensor([mnist_train_full[i][1] for i in range(len(mnist_train_full))])

    print(f"Loaded MNIST subset with {len(data):,} images.")
    print(f"Image tensor shape: {data.shape}")
    print(f"Labels tensor shape: {targets.shape}")

    return data, targets

images, targets = load_mnist(fraction=1.)
# Define split ratio
validation_ratio = 0.25  # TODO
# Perform the split with stratification to maintain label balance
X_train, X_val, y_train, y_val = train_test_split(images, targets, test_size=validation_ratio, stratify=targets,
                                                  random_state=42)
# Set as TensorDatasets
ratio = 0.1  # reduce training data
train_dataset = TensorDataset(X_train[:int(len(X_train) * ratio)], y_train[:int(len(X_train) * ratio)])
val_dataset = TensorDataset(X_val, y_val)
# Check the size of the splits
print(f"Train set: {len(train_dataset)} samples")
print(f"Validation set: {len(val_dataset)} samples")


# 1. Define the LightningModule
class LitMNIST(pl.LightningModule):
    def __init__(self, hidden=128, layers=2, lr=1e-3, dropout=0., optimizer='Adam'):
        super().__init__()
        # save hyperparameters
        self.save_hyperparameters()

        # set architecture
        model = [torch.nn.Flatten(),
                 torch.nn.Linear(28 * 28, hidden),
                 torch.nn.ReLU(),
                 torch.nn.Dropout(p=dropout)]  # Dropout after each ReLU

        for i in range(layers - 2):
            model.append(torch.nn.Linear(hidden, hidden))
            model.append(torch.nn.ReLU())
            model.append(torch.nn.Dropout(p=dropout))  # Dropout after each ReLU

        model.append(torch.nn.Linear(hidden, 10))  # Final output layer without dropout

        self.model = torch.nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        if self.hparams.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer


def train(config=None):
    # W&B provides with a configuration from the configuration space
    with wandb.init(config=config):
        args = wandb.config

    # Logger
    wandb_logger = WandbLogger(project="mnist-sweep",
                               name=f"layers={args.layers}_hidden={args.hidden}_lr={args.lr:.0e}_Opt={args.optimizer}",
                               log_model=False)

    # Set seed
    pl.seed_everything(52, workers=True)

    # Data Loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    # Model instantiation
    model = LitMNIST(hidden=args.hidden, layers=args.layers, lr=args.lr, dropout=args.dropout,
                     optimizer=args.optimizer)

    # save best trained model
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1)

    # Trainer
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        max_epochs=5,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        deterministic=True,
        num_sanity_val_steps=0
    )

    # Fit
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':

    sweep_config = {
        "method": "random",  # G1: "random" | G2: "grid" | G3: "bayes" | G4: "student-search"
        "metric": {
            "name": "val_acc",
            "goal": "maximize"
        },
        "parameters": {
            "lr": {
                "values": [1e-3]
            },
            "hidden": {
                "values": [1024, 2048]
            },
            "dropout": {
                "values": [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.45, 0.5]
            },
            "layers": {
                "values": [4, 8, 16]
            },
            "optimizer": {
                "values": ["Adam"]
            }
        }
    }

    # Create the sweep
    sweep_id = wandb.sweep(sweep_config, project="mnist-sweep")

    # Run the agent
    wandb.agent(sweep_id, function=train)