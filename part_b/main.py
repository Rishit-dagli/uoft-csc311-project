import argparse

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from lion_pytorch import Lion


class Attention(nn.Module):
    """
    We do not do this anymore and is obsolete now.
    """

    def __init__(self, embed_size, units):
        super(Attention, self).__init__()
        self.W1 = nn.Linear(embed_size, units)
        self.W2 = nn.Linear(embed_size, units)
        self.V = nn.Linear(units, 1)

    def forward(self, question_emb, user_emb):
        q = self.W1(question_emb)
        u = self.W2(user_emb)
        combined = torch.tanh(q + u)
        attention_weights = torch.softmax(self.V(combined), dim=1)
        context_vector = attention_weights * question_emb
        context_vector = torch.sum(context_vector, dim=1)
        return context_vector


class CustomLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(CustomLoss, self).__init__()
        self.delta = delta

    def forward(self, predictions, targets):
        error = targets - predictions
        is_small_error = torch.abs(error) < self.delta
        small_error_loss = 0.5 * torch.pow(error, 2)
        large_error_loss = self.delta * torch.abs(error) - 0.5 * self.delta**2
        loss = torch.where(is_small_error, small_error_loss, large_error_loss)
        return torch.mean(loss)


class CustomDataset(Dataset):
    def __init__(self, filepath):
        self.data = pd.read_csv(filepath)
        self.question_ids = self.data["question_id"].to_numpy()
        self.user_ids = self.data["user_id"].to_numpy()
        self.labels = self.data["is_correct"].to_numpy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "question_id": self.question_ids[idx],
            "user_id": self.user_ids[idx],
            "label": self.labels[idx],
        }


class QuestionUserModel(pl.LightningModule):
    def __init__(
        self,
        num_questions,
        num_users,
        customloss,
        embedding_dim=16,
        model_type="original",
        optimizer="adam",
        **kwargs,
    ):
        super().__init__()
        self.model_type = model_type
        self.optimizer = optimizer
        self.question_embedding = nn.Embedding(num_questions, embedding_dim)
        self.user_embedding = nn.Embedding(num_users, embedding_dim)

        if model_type == "mhsa":
            self.attention = nn.MultiheadAttention(
                embed_dim=2 * embedding_dim, num_heads=4
            )
            self.fc = nn.Sequential(
                nn.Linear(2 * embedding_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(embedding_dim * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )
        self.accuracy = torchmetrics.Accuracy(task="binary")
        if customloss:
            self.loss_fn = CustomLoss()
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()
        self.save_hyperparameters()

    def forward(self, question_id, user_id):
        question_embed = self.question_embedding(question_id)
        user_embed = self.user_embedding(user_id)

        if self.model_type == "mhsa":
            # Reshape embeddings to match (L, N, E) format expected by MultiHeadAttention
            # In this case, L = sequence length (1 for single combined embedding), N = batch size, E = embedding dimension

            combined_embed = torch.cat((question_embed, user_embed), dim=-1).unsqueeze(
                0
            )
            attn_output, _ = self.attention(
                combined_embed, combined_embed, combined_embed
            )
            x = attn_output.squeeze(
                0
            )  # Remove sequence length dimension after attention
        else:
            x = torch.cat((question_embed, user_embed), dim=-1)
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        question_id, user_id, labels = (
            batch["question_id"],
            batch["user_id"],
            batch["label"],
        )
        labels = labels.float().unsqueeze(1)
        predictions = self(question_id, user_id)
        loss = self.loss_fn(predictions, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        question_id, user_id, labels = (
            batch["question_id"],
            batch["user_id"],
            batch["label"],
        )
        labels = labels.float().unsqueeze(1)
        predictions = self(question_id, user_id)
        loss = self.loss_fn(predictions, labels)
        preds = torch.round(torch.sigmoid(predictions))
        self.log("val_loss", loss, prog_bar=True)
        self.accuracy(preds, labels.int())
        self.log("val_acc", self.accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = None
        if self.optimizer == "adam":
            optimizer = Adam(self.parameters(), lr=1e-3)
        elif self.optimizer == "lion":
            optimizer = Lion(self.parameters(), lr=1e-3, weight_decay=1e-2, use_triton=True)
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=1e-2, momentum=0.9)
        elif self.optimizer == "rmsprop":
            optimizer = torch.optim.RMSprop(self.parameters(), lr=1e-2)
        elif self.optimizer == "adagrad":
            optimizer = torch.optim.Adagrad(self.parameters(), lr=1e-2)
        elif self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=1e-2)
        elif self.optimizer == "adadelta":
            optimizer = torch.optim.Adadelta(self.parameters(), lr=1.0)
        elif self.optimizer == "adamax":
            optimizer = torch.optim.Adamax(self.parameters(), lr=2e-3)
        else:
            raise ValueError(f"Invalid optimizer: {self.optimizer}")
        if self.hparams.lr_scheduler == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma
            )
        elif self.hparams.lr_scheduler == "cosineannealing":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams.step_size, eta_min=0
            )
        elif self.hparams.lr_scheduler == "reducelronplateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                "min",
                patience=self.hparams.patience,
                factor=self.hparams.gamma,
            )
        elif self.hparams.lr_scheduler == "cosinedecay":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams.epochs, eta_min=0
            )
        else:
            return optimizer
        if self.hparams.lr_scheduler == "reducelronplateau":
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                },
            }
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "interval": "epoch",
            }

    def test_step(self, batch, batch_idx):
        question_id, user_id, labels = (
            batch["question_id"],
            batch["user_id"],
            batch["label"],
        )
        labels = labels.float().unsqueeze(1)
        predictions = self(question_id, user_id)
        loss = self.loss_fn(predictions, labels)
        preds = torch.sigmoid(predictions)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        np.savez("test_preds.npz", preds=torch.round(preds).detach().cpu().numpy())
        np.savez("test_labels.npz", labels=labels.detach().cpu().numpy())
        self.accuracy(preds, labels.int())
        self.log("test_acc", self.accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss


class DataModule(pl.LightningDataModule):
    def __init__(self, train_file, val_file, test_file, batch_size):
        super().__init__()
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = CustomDataset(self.train_file)
        self.val_dataset = CustomDataset(self.val_file)
        self.test_dataset = CustomDataset(self.test_file)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


def train_model(
    train_file,
    val_file,
    test_file,
    batch_size=32,
    epochs=10,
    device="cpu",
    project_name="test",
    num_devices=1,
    checkpoint_path=None,
    save_model_path="model.pth",
    custom_loss=True,
    optimizer="adam",
    lr_scheduler="none",
    step_size=10,
    gamma=0.1,
    patience=10,
    lr=1e-3,
    model_type="original",
):
    max_question_id = 1773
    max_user_id = 541

    model = QuestionUserModel(
        num_questions=max_question_id + 1,
        num_users=max_user_id + 1,
        customloss=custom_loss,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        step_size=step_size,
        gamma=gamma,
        patience=patience,
        lr=lr,
        model_type=model_type,
    )

    data_module = DataModule(train_file, val_file, test_file, batch_size)

    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")

    logger = TensorBoardLogger("logs", name=project_name)

    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[checkpoint_callback],
        accelerator=device,
        logger=logger,
        devices=num_devices,
    )

    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path))
    trainer.fit(model, data_module)

    if save_model_path:
        torch.save(model.state_dict(), save_model_path)
    data_module.setup(stage="test")
    trainer.test(model, datamodule=data_module)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Model for educational data.")
    parser.add_argument(
        "--project", type=str, default="test", help="Name of the project."
    )
    parser.add_argument(
        "--custom_loss", action="store_true", help="Use custom loss function."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="original",
        choices=["original", "mhsa"],
        help="Model architecture type: 'original' or 'mhsa'.",
    )

    parser.add_argument(
        "--optimizer", type=str, default="adam", help="Optimizer to use."
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="none",
        choices=[
            "none",
            "steplr",
            "cosineannealing",
            "reducelronplateau",
            "cosinedecay",
        ],
        help="Type of LR scheduler: steplr, cosineannealing, reducelronplateau, cosinedecay, none.",
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=10,
        help="Step size for StepLR and T_max for CosineAnnealing.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="Gamma for StepLR, ReduceLROnPlateau, and factor for CosineAnnealing.",
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Patience for ReduceLROnPlateau."
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate.")

    parser.add_argument(
        "--base_path", type=str, default="../../data", help="Base path for the dataset."
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training.")
    parser.add_argument(
        "--num_devices",
        type=int,
        default=1,
        help="Number of devices to use for training.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--save_model", type=str, default="model.pth", help="Path to save the model."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from.",
    )
    parser.add_argument("--seed", type=int, default=3047, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    train_file = f"{args.base_path}/train_data.csv"
    valid_file = f"{args.base_path}/valid_data.csv"
    test_file = f"{args.base_path}/test_data.csv"
    device = "cpu"
    if args.gpu:
        assert torch.cuda.is_available(), "GPU is not available."
        device = "cuda"
    pl.seed_everything(args.seed)

    train_model(
        train_file,
        valid_file,
        test_file,
        args.batch_size,
        args.epochs,
        device,
        args.project,
        args.num_devices,
        args.checkpoint_path,
        args.save_model,
        args.custom_loss,
        args.optimizer,
        args.lr_scheduler,
        args.step_size,
        args.gamma,
        args.patience,
        args.lr,
        args.model_type,
    )


if __name__ == "__main__":
    main()
