import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from utils import *


def load_data(base_path):
    """
    Load the data in PyTorch Tensor.
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    zero_train_matrix[np.isnan(train_matrix)] = 0
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100, dropout_rate=0.5):
        super(AutoEncoder, self).__init__()
        self.g = nn.Linear(num_question, k)
        self.dropout = nn.Dropout(dropout_rate)  # Define dropout layer
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        x = torch.sigmoid(self.g(inputs))
        x = self.dropout(x)
        out = torch.sigmoid(self.h(x))
        return out


def criterion(output, target, delta):
    abs_error = torch.abs(output - target)
    quadratic = torch.min(abs_error, torch.tensor([delta]).to(abs_error.device))
    linear = abs_error - quadratic
    loss = 0.5 * quadratic**2 + delta * linear
    return torch.sum(loss)


def train(
    model,
    lr,
    lamb,
    delta,
    train_data,
    zero_train_data,
    valid_data,
    num_epoch,
    verbose=False,
    device="cuda:0",
):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    validation_accuracy = []
    training_loss = []
    validation_loss = []

    for epoch in range(num_epoch):
        train_loss = 0.0

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            inputs = inputs.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(inputs)

            reg_term = 0.5 * lamb * model.get_weight_norm()

            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0].unsqueeze(0)[nan_mask] = output[0].unsqueeze(0)[nan_mask]

            loss = criterion(output, target, delta) + reg_term
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data, device=device)

        model.eval()
        valid_loss = 0.0
        for i, u in enumerate(valid_data["user_id"]):
            inputs = Variable(zero_train_data[u]).unsqueeze(0)
            output = model(inputs)
            guess = output[0][valid_data["question_id"][i]].item()
            valid_loss += criterion(
                output[0][valid_data["question_id"][i]],
                torch.tensor([valid_data["is_correct"][i]]).to(device),
                delta,
            )

        if verbose:
            print(
                f"Epoch: {epoch} \tTraining Cost: {train_loss:.6f}\t Valid Acc: {valid_acc}"
            )
        training_loss.append(train_loss)
        validation_loss.append(valid_loss.item())  # Ensure this is a scalar
        validation_accuracy.append(valid_acc)

    return training_loss, validation_loss, validation_accuracy


def evaluate(model, train_data, valid_data, device):
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0).to(device)
        output = model(inputs).to(device)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an AutoEncoder for educational data."
    )
    parser.add_argument(
        "--base_path", type=str, default="../../data", help="Base path for the dataset."
    )
    parser.add_argument(
        "--k", type=int, default=16, help="Dimensionality of the hidden layer."
    )
    parser.add_argument("--lr", type=float, default=0.03, help="Learning rate.")
    parser.add_argument(
        "--lamb", type=float, default=0.0, help="Regularization parameter."
    )
    parser.add_argument("--num_epoch", type=int, default=30, help="Number of epochs.")
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed logging information."
    )
    parser.add_argument("--seed", type=int, default=3047, help="Random seed.")
    parser.add_argument(
        "--save_model", type=str, default="model.pth", help="Path to save the model."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="losscurves.png",
        help="Path to save the loss curves.",
    )
    parser.add_argument("--dropout_rate", type=float, default=0.0, help="Dropout rate.")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for training.")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        "cuda" if args.use_gpu and torch.cuda.is_available() else "cpu"
    )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    zero_train_matrix, train_matrix, valid_data, test_data = load_data(args.base_path)

    model = AutoEncoder(
        num_question=train_matrix.shape[1], k=args.k, dropout_rate=args.dropout_rate
    ).to(device)

    train_losses, valid_losses, valid_acc = train(
        model,
        args.lr,
        args.lamb,
        0.2,
        train_matrix,
        zero_train_matrix,
        valid_data,
        args.num_epoch,
        verbose=args.verbose,
        device=device,
    )
    torch.save(model.state_dict(), args.save_model)

    N_valid = sum(
        [1 for is_correct in valid_data["is_correct"] if is_correct is not None]
    )
    N_train = np.count_nonzero(~np.isnan(train_matrix.numpy()))

    epochs = range(1, args.num_epoch + 1)
    plt.figure()
    plt.plot(epochs, np.array(train_losses) / N_train, label="Training loss")
    plt.plot(epochs, np.array(valid_losses) / N_valid, label="Validation loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    if args.save_path is not None:
        plt.savefig(args.save_path)


if __name__ == "__main__":
    main()
