from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import matplotlib.pyplot as plt

import numpy as np
import torch


def load_data(base_path="../data"):
    """Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        x = torch.sigmoid(self.g(inputs))
        out = torch.sigmoid(self.h(x))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(
    model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch, verbose=False
):
    """Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function.

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    validation_accuracy = []
    training_loss = []
    validation_loss = []

    for epoch in range(0, num_epoch):
        train_loss = 0.0
        correct = 0.0
        total = 0.0

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            reg_term = lamb * model.get_weight_norm()

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0].unsqueeze(0)[nan_mask] = output[0].unsqueeze(0)[nan_mask]

            loss = torch.sum((output - target) ** 2.0 + reg_term)
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)

        model.eval()
        valid_loss = 0.0
        for i, u in enumerate(valid_data["user_id"]):
            inputs = Variable(zero_train_data[u]).unsqueeze(0)
            output = model(inputs)
            guess = output[0][valid_data["question_id"][i]].item()
            valid_loss += (guess - valid_data["is_correct"][i]) ** 2

        if verbose:
            print(
                "Epoch: {} \tTraining Cost: {:.6f}\t "
                "Valid Acc: {}".format(epoch, train_loss, valid_acc)
            )
        training_loss.append(train_loss)
        validation_loss.append(valid_loss)
        validation_accuracy.append(valid_acc)
    return training_loss, validation_loss, validation_accuracy
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################

    print("Start hyperparameter tuning for part (b).")
    print()

    k_values = [10, 50, 100, 200, 500]
    lr_values = [0.01, 0.03, 0.05]
    epoch_values = [10, 20]
    lamb = 0.0

    best_config = {
        "k": k_values[0],
        "lr": lr_values[0],
        "epochs": epoch_values[0],
        "valid_acc": -float("inf"),
    }

    for k in k_values:
        for lr in lr_values:
            for num_epoch in epoch_values:
                print(f"Training with k={k}, lr={lr}, epochs={num_epoch}")

                # Initialize model
                model = AutoEncoder(num_question=train_matrix.shape[1], k=k)

                # Train the model
                train(
                    model,
                    lr,
                    lamb,
                    train_matrix,
                    zero_train_matrix,
                    valid_data,
                    num_epoch,
                )

                valid_acc = evaluate(model, zero_train_matrix, valid_data)

                # print(f"Validation accuracy: {valid_acc}")

                if valid_acc > best_config["valid_acc"]:
                    best_config.update(
                        {"k": k, "lr": lr, "epochs": num_epoch, "valid_acc": valid_acc}
                    )

    print(
        f"Best configuration: k={best_config['k']}, lr={best_config['lr']}, epochs={best_config['epochs']} with validation accuracy: {best_config['valid_acc']}"
    )

    print()
    print("Start part (c)")
    print()

    N_valid = np.count_nonzero(valid_data["is_correct"])
    N_train = np.count_nonzero(zero_train_matrix)

    model = AutoEncoder(num_question=train_matrix.shape[1], k=best_config["k"])
    lamb = 0.0
    train_losses, valid_losses, valid_acc = train(
        model,
        best_config["lr"],
        lamb,
        train_matrix,
        zero_train_matrix,
        valid_data,
        best_config["epochs"],
    )

    test_accuracy = evaluate(model, zero_train_matrix, test_data)
    print(f"Test accuracy: {test_accuracy}")

    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, np.array(train_losses) / N_train, label="Training loss")
    plt.plot(epochs, np.array(valid_losses) / N_valid, label="Validation loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("part3c.png")

    print("Start hyperparameter tuning for part (d)")
    print()

    lamb = [1e-3, 1e-2, 1e-1, 1]
    best_config_l = {
        "lamb": lamb[0],
        "valid_acc": -float("inf"),
    }

    for l in lamb:
        print(f"Training with lambda={l}")

        model = AutoEncoder(num_question=train_matrix.shape[1], k=best_config["k"])
        train(
            model,
            best_config["lr"],
            l,
            train_matrix,
            zero_train_matrix,
            valid_data,
            best_config["epochs"],
        )

        valid_acc = evaluate(model, zero_train_matrix, valid_data)

        if valid_acc > best_config_l["valid_acc"]:
            best_config_l.update({"lamb": l, "valid_acc": valid_acc})

    model = AutoEncoder(num_question=train_matrix.shape[1], k=best_config["k"])
    train(
        model,
        best_config["lr"],
        best_config_l["lamb"],
        train_matrix,
        zero_train_matrix,
        valid_data,
        best_config["epochs"],
    )
    test_accuracy = evaluate(model, zero_train_matrix, test_data)
    print(
        f"Best configuration: k={best_config['k']}, lr={best_config['lr']}, epochs={best_config['epochs']}, lambda={best_config_l['lamb']} with validation accuracy: {best_config_l['valid_acc']} and test accuracy: {test_accuracy}"
    )
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
