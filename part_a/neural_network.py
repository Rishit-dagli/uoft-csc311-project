import pandas as pd
from matplotlib import pyplot as plt

from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.utils.data

import numpy as np
import torch
from scipy.sparse import *


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

    valid_data_df = load_valid_csv_to_df(base_path)
    test_data_df = load_public_test_csv_to_df(base_path)
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    student_meta = pd.read_csv(
        os.path.join(base_path, 'student_meta_clean.csv'))
    question_meta = pd.read_csv(os.path.join(base_path, 'question_meta.csv'))
    subject_meta = pd.read_csv(os.path.join(base_path, 'subject_meta.csv'))

    student_meta_cleaned = process_student_metadata(student_meta)
    question_meta_cleaned = process_question_metadata(question_meta,
                                                      subject_meta)

    train_data = load_train_csv_df(base_path)



    # perf matrix is train_matrix without metadata
    perf_matrix = load_train_sparse(base_path).toarray() #ndarray
    zero_perf_matrix = perf_matrix.copy()
    zero_perf_matrix[np.isnan(perf_matrix)] = 0
    zero_perf_matrix = torch.FloatTensor(zero_perf_matrix)
    perf_matrix = torch.FloatTensor(perf_matrix)

    combined_train_matrix = combine_metadata(train_data,
                                             student_meta_cleaned,
                                             question_meta_cleaned).toarray()

    zero_train_matrix = combined_train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[pd.isnull(combined_train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(combined_train_matrix)
    return (zero_train_matrix, train_matrix, valid_data,
            test_data, zero_perf_matrix, perf_matrix)


def combine_metadata(train_data, student_meta, question_meta):
    """
    Combine the student and question metadata with the given sparse dataset.

    :param data: DF representing the training data.
    :param student_meta: DataFrame containing processed student metadata.
    :param question_meta: DataFrame containing processed question metadata.
    :return: Combined sparse dataset.
    """
    # Convert metadata DataFrames to sparse format
    sparse_matrix = data_df_to_sparse(train_data)
    # Append metadata to the sparse data
    question_meta_sparse = (csc_matrix(question_meta.values)).transpose()


    # Convert the student metadata DataFrame to a sparse matrix
    student_meta_sparse = csc_matrix(student_meta.values)
    print(student_meta_sparse.shape)  # Output: (6, 542)
    student_meta_sparse = vstack([np.full((290, 6), np.nan), student_meta_sparse], format='csc')

    # Concatenate the question metadata sparse matrix on top of the original sparse matrix
    combined_matrix = vstack([question_meta_sparse, sparse_matrix], format='csc')
    print(combined_matrix.shape)  # Output: (2067, 542

    # Concatenate the student metadata sparse matrix to the left of the combined matrix
    final_matrix = hstack([combined_matrix, student_meta_sparse], format='csc')
    final_matrix = final_matrix
    print(final_matrix.shape)  # Output: (542, 2070)

    return final_matrix


class AutoEncoder(nn.Module):
    def __init__(self, input_size, num_question, k=100):
        """Initialize a class AutoEncoder.

        :param input_size: int
        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        self.g = nn.Sequential(
            nn.Linear(input_size, k),
            nn.ReLU(),
        )
        self.h = nn.Sequential(
            nn.Linear(k, num_question),
            nn.ReLU(),
        )

    def get_weight_norm(self):
        """Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, x):
        x = self.g(x)
        x = self.h(x)
        return x


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch,
          zero_perf_matrix, perf_matrix, verbose=False):
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
    # Tell PyTorch you are training the model.
    model.train()

    criterion = nn.MSELoss()
    # Define optimizers and loss function.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_student = train_data.shape[0]
    validation_accuracy = []
    training_loss = []
    validation_loss = []

    for epoch in range(0, num_epoch):
        train_loss = 0.0

        # for user_id in range(num_student):
        #     inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
        #     target = Variable(zero_perf_matrix[user_id]).unsqueeze(0)
        #
        #     optimizer.zero_grad()
        #     output = model(inputs)
        #
        #     # reg_term = lamb / 2 * model.get_weight_norm()
        #
        #     # Mask the target to only compute the gradient of valid entries.
        #     nan_mask = np.isnan(perf_matrix[user_id].unsqueeze(0).numpy())
        #     target[0].unsqueeze(0)[nan_mask] = output[0].unsqueeze(0)[nan_mask]
        #
        #     loss = criterion(output, target) #+ reg_term
        #     loss.backward()
        #
        #     train_loss += loss.item()
        #     optimizer.step()
        inputs = zero_train_data
        target = zero_perf_matrix

        output = model(inputs)
        # Assuming 'output' and 'target' are your existing tensors
        output_size = output.shape[0]
        target_size = target.shape[0]

        # Calculate the amount of padding needed
        padding_needed = output_size - target_size
        nan_padding = torch.full((padding_needed, target.shape[1]),
                                 float('nan'))

        # Concatenate the padding with the target tensor
        padded_target = torch.cat((target, nan_padding), dim=0)
        nan_mask = np.isnan(padded_target.numpy())
        padded_target[nan_mask] = output[nan_mask]
        optimizer.zero_grad()
        loss = criterion(output, padded_target)
        loss.backward()
        optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
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
    zero_train_matrix, train_matrix, valid_data, test_data, perf_matrix, zero_perf_matrix = load_data()

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
                model = AutoEncoder(input_size=train_matrix.shape[1],
                                    num_question=perf_matrix.shape[1],
                                    k=k)

                # Train the model
                train(
                    model,
                    lr,
                    lamb,
                    train_matrix,
                    zero_train_matrix,
                    valid_data,
                    num_epoch,
                    zero_perf_matrix,
                    perf_matrix
                )

                valid_acc = evaluate(model, zero_train_matrix, valid_data)

                # print(f"Validation accuracy: {valid_acc}")

                if valid_acc > best_config["valid_acc"]:
                    best_config.update(
                        {"k": k, "lr": lr, "epochs": num_epoch,
                         "valid_acc": valid_acc}
                    )

    print(
        f"Best configuration: k={best_config['k']}, lr={best_config['lr']}, epochs={best_config['epochs']} with validation accuracy: {best_config['valid_acc']}"
    )

    print()
    print("Start part (c)")
    print()

    N_valid = np.count_nonzero(valid_data["is_correct"])
    N_train = np.count_nonzero(zero_train_matrix)

    model = AutoEncoder(input_size=train_matrix.shape[1],
                        num_question=perf_matrix.shape[1],
                        k=best_config["k"])
    lamb = 0.0
    train_losses, valid_losses, valid_acc = train(
        model,
        best_config["lr"],
        lamb,
        train_matrix,
        zero_train_matrix,
        valid_data,
        best_config["epochs"],
        zero_perf_matrix,
        perf_matrix
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

        model = AutoEncoder(input_size=train_matrix.shape[1],
                            num_question=perf_matrix.shape[1],
                            k=best_config["k"])
        train(
            model,
            best_config["lr"],
            l,
            train_matrix,
            zero_train_matrix,
            valid_data,
            best_config["epochs"],
            zero_perf_matrix,
            perf_matrix
        )

        valid_acc = evaluate(model, zero_train_matrix, valid_data)

        if valid_acc > best_config_l["valid_acc"]:
            best_config_l.update({"lamb": l, "valid_acc": valid_acc})

    model = AutoEncoder(input_size=train_matrix.shape[1],
                        num_question=perf_matrix.shape[1],
                        k=best_config["k"])
    train(
        model,
        best_config["lr"],
        best_config_l["lamb"],
        train_matrix,
        zero_train_matrix,
        valid_data,
        best_config["epochs"],
        zero_perf_matrix,
        perf_matrix
    )
    test_accuracy = evaluate(model, zero_train_matrix, test_data)
    print(
        f"Best configuration: k={best_config['k']}, lr={best_config['lr']}, epochs={best_config['epochs']}, lambda={best_config_l['lamb']} with validation accuracy: {best_config_l['valid_acc']} and test accuracy: {test_accuracy}"
    )


if __name__ == "__main__":
    main()
