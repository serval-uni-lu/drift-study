import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from constrained_attacks.attacks.cta.cfab import CFAB
from mlc.constraints.constraints import Constraints
from mlc.transformers.tab_scaler import TabScaler
from torch import nn


def attack(model, scaler: TabScaler, x, label):
    constraints = Constraints(
        feature_types=np.array(["real"] * x.shape[1]),
        mutable_features=np.array([True] * x.shape[1]),
        lower_bounds=scaler.x_min,
        upper_bounds=scaler.x_max,
        relation_constraints=None,
        feature_names=None,
    )

    attack = CFAB(
        constraints,
        scaler,
        model.wrapper_model,
        model.predict_proba,
        norm="L2",
        eps=10,
        steps=30,
        n_restarts=5,
        alpha_max=0.1,
        eta=1.05,
        beta=0.9,
        verbose=False,
        seed=0,
        multi_targeted=False,
        n_classes=10,
        fix_equality_constraints_end=False,
        fix_equality_constraints_iter=False,
        eps_margin=0.01,
    )

    return attack(x, label)


def distance(scaler, x1, x2):
    return np.linalg.norm(scaler.transform(x1) - scaler.transform(x2), axis=1)


def window_mean(arr, win):
    arr.copy()
    rest = arr.shape[0] % win
    if rest != 0:
        arr = arr[:-rest]
    arr = arr.reshape(-1, win)
    return arr.mean(1)


def plot_correlation_time_scale(list_x, list_label):
    if len(list_x) != len(list_label):
        raise ValueError("list_x and list_label must have the same length")
    if len(list_x) != 2:
        raise ValueError("list_x must have length 2")

    curve_1 = list_label[0]
    curve_2 = list_label[1]
    sample = 50

    plt.figure(figsize=(10, 6))
    ax1 = sns.lineplot(list_x[0], color="blue", label=curve_1)
    ax1.set_ylabel(curve_1, color="blue")

    # Create the second plot (right y-axis)
    ax2 = ax1.twinx()
    sns.lineplot(list_x[1], color="red", ax=ax2, label=curve_2)
    ax2.set_ylabel(curve_2, color="red")

    # Add a legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper left")
    plt.show()


def plot_correlation_time(list_x, list_label):
    df = pd.DataFrame({list_label[i]: list_x[i] for i in range(len(list_x))})
    df["index"] = np.arange(len(df))
    for i in range(len(list_x)):
        sns.lineplot(data=df, x="index", y=list_x[i])
    plt.show()


def plot_correlation_scatter(list_x, list_label):
    if len(list_x) != len(list_label):
        raise ValueError("list_x and list_label must have the same length")
    if len(list_x) != 2:
        raise ValueError("list_x must have length 2")

    df = pd.DataFrame({list_label[i]: list_x[i] for i in range(len(list_x))})

    sns.scatterplot(data=df, x=np.array(list_x[0]), y=list_x[1])
    plt.show()


def success_rate(model, x, y):
    y_pred = model.predict(x)

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()

    return (y_pred != y).mean()


def analyse(
    model,
    scaler,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    x_adv_pred,
    x_adv_test,
):
    y_pred = model.predict(x_test.cpu().numpy())
    y_pred_correct = y_pred == y_test.numpy()
    if x_adv_pred is None:
        x_adv_pred = (
            attack(model, scaler, x_test, torch.from_numpy(y_pred).to("cuda"))
            .detach()
            .cpu()
            .numpy()
        )
    if x_adv_test is None:
        x_adv_test = (
            attack(model, scaler, x_test, y_test).detach().cpu().numpy()
        )

    print("success rate pred: ", success_rate(model, x_adv_pred, y_pred))
    print("success rate test: ", success_rate(model, x_adv_test, y_test))
    WIN = 20000
    acc_win = window_mean(y_pred_correct, WIN)

    MOVE = 5

    loss = nn.CrossEntropyLoss(reduction="none")
    y_score = torch.from_numpy(
        model.predict_proba(x_test.cpu().numpy())
    ).float()
    y_loss_test = loss(y_score, y_test).detach().cpu().numpy()
    y_loss_test_win = window_mean(y_loss_test, WIN)

    for x_adv in [x_adv_pred]:
        x_dist = distance(scaler, x_test.cpu().numpy(), x_adv)
        x_dist_win = window_mean(x_dist, WIN)

        x_dist_win_delta = x_dist_win[1:] - x_dist_win[:-1]
        acc_win_delta = acc_win[1:] - acc_win[:-1]
        y_loss_test_win_delta = y_loss_test_win[1:] - y_loss_test_win[:-1]

        acc_win_time_shift = acc_win[MOVE:]
        x_dist_win_time_shift = x_dist_win[:-MOVE]

        plot_correlation_time_scale(
            [x_dist_win, y_loss_test_win], ["dist", "y_loss_test_win"]
        )
        # plot_correlation_time_scale([x_dist_win_delta, acc_win_delta], ["dist_delta", "acc_delta"])

        plot_correlation_scatter(
            [x_dist_win, acc_win], ["dist", "y_loss_test_win"]
        )
        print(np.corrcoef(x_dist_win, y_loss_test_win))
        plot_correlation_scatter(
            [x_dist_win_delta, acc_win_delta],
            ["dist_delta", "y_loss_test_win_delta"],
        )
        print(np.corrcoef(x_dist_win_delta, y_loss_test_win_delta))

        # plot_correlation_scatter([x_dist_win_time_shift, acc_win_time_shift], ["dist", "acc"])
        # print(np.corrcoef(x_dist_win_time_shift, acc_win_time_shift))

        # print("Correlation loss and acc: ", np.corrcoef(y_loss_test, x_dist))

    return x_adv_pred, x_adv_test


def analyse_loss(
    model,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
):
    y_score = torch.from_numpy(
        model.predict_proba(x_test.cpu().numpy())
    ).float()
    loss = nn.CrossEntropyLoss(reduction="none")
    y_pred = torch.argmax(y_score, 1)
    y_loss = loss(y_score, y_pred).detach().cpu().numpy()

    y_pred_correct = y_pred.numpy() == y_test.numpy()

    WIN = 5000
    acc_win = window_mean(y_pred_correct, WIN)
    loss_win = window_mean(y_loss, WIN)

    loss_win_delta = loss_win[1:] - loss_win[:-1]
    acc_win_delta = acc_win[1:] - acc_win[:-1]

    plot_correlation_scatter([loss_win, acc_win], ["loss", "acc"])
    print(np.corrcoef(loss_win, acc_win))
    plot_correlation_scatter(
        [loss_win_delta, acc_win_delta], ["loss_delta", "acc_delta"]
    )
    print(np.corrcoef(loss_win_delta, acc_win_delta))
