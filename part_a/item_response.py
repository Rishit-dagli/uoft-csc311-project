from matplotlib import pyplot as plt

from utils import *

import numpy as np

from part_a.utils import load_public_test_csv, load_train_csv, \
    load_train_sparse, load_valid_csv


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    neg_lld = 0.
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        p_a = sigmoid(theta[u] - beta[q])
        c = data["is_correct"][i]
        neg_lld += c * np.log(p_a) + (1 - c) * np.log(1 - p_a)
    return -neg_lld


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    new_beta, new_theta = beta.copy(), theta.copy()
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        c = data["is_correct"][i]
        p_a = sigmoid(theta[u] - beta[q])
        new_theta[u] -= lr * (p_a - c)
        new_beta[q] -= lr * (c - p_a)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return new_theta, new_beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    student_ids = data["user_id"]
    question_ids = data["question_id"]

    unique_student_ids = np.unique(student_ids)
    unique_question_ids = np.unique(question_ids)

    beta = {q: np.random.normal(0, 1) for q in unique_question_ids}
    theta = {u: np.random.normal(0, 1) for u in unique_student_ids}

    val_acc_lst = []
    train_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        train_acc = evaluate(data, theta=theta, beta=beta)
        print("NLLK: {} \t  Train Acc: {}".format(neg_lld, train_acc))
        train_acc_lst.append(train_acc)
        val_score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(val_score)
        print("NLLK: {} \t  Valid Acc: {}".format(neg_lld, val_score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    return theta, beta, val_acc_lst, train_acc_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = theta[u] - beta[q]
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")

    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    lr = 0.01
    iterations = 35
    theta, beta, val_acc_lst, train_acc_lst = irt(train_data, val_data, lr, iterations)


    # Plot the validation accuracy over iterations (list_length),
    # training accuracy over iterations on the same plot but in red, put a legend

    fig, ax = plt.subplots()
    ax.plot(range(iterations), val_acc_lst
            ,label='validation accuracy')
    ax.plot(range(iterations), train_acc_lst
            ,label='training accuracy', color='r')
    ax.legend()
    ax.xaxis.set_label_text('iterations')
    ax.yaxis.set_label_text('accuracy')
    ax.set_title(
        'validation accuracy over iterations')
    plt.savefig('./2c.png')
    # Best validation accuracy:  0.7064634490544736
    # Test accuracy:  0.7092859158904883
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################                                                          #
    # Implement part (d). Select 3 questions, for each, plot a curve showing
    # the P of the correct response as fn of the ability theta using the
    # trained theta, beta. put all three curves on the same figure.#
    #####################################################################

    q_list = [0,1,2]
    plot_legends = []
    for i in range(3):
        plot_legends.append(f'question_id: {q_list[i]}')

    theta_range = np.linspace(-3, 3, num=101)

    curve_colors = ['r', 'g', 'b']
    fig, ax = plt.subplots()

    for i in range(3):
        q = q_list[i]
        # list of probabilities p(c_uq) for each theta given question q
        prob_list = sigmoid(theta_range - beta[q])
        # plot p(c_uq) as a function of theta given question q
        ax.plot(theta_range, prob_list, curve_colors[i])

    ax.xaxis.set_label_text('theta')
    ax.yaxis.set_label_text('p(c_ij)')
    ax.set_title(
        'p(c_ij) as a function of theta given three different questions')
    ax.legend(plot_legends)
    plt.savefig(
        './2d.png')

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
