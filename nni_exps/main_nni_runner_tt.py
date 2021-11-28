import sys

sys.path.insert(0, "..")

from microbiome2matrix import augment, seperate_according_to_tag, otu22d, dendogram_ordering
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score, f1_score, multilabel_confusion_matrix, \
    confusion_matrix
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import GroupShuffleSplit
from pytorch_lightning.callbacks import EarlyStopping
from Naieve_model.naeive_model import Naeive
# from captum.attr import IntegratedGradients
from CNN_model.CNN1convlayer import CNN_1l
from CNN_model.CNN_D3 import CNN_D3
from torch.utils import data as data_modul
from nni_data_loader import load_nni_data
from CNN_model.CNN2convlayer import CNN
from collections import defaultdict
from scipy.stats import spearmanr
from itertools import zip_longest
from numpy.fft import fft2

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import random
import torch
import nni

SEED = None


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed, workers=True)
    torch.use_deterministic_algorithms(True)
    global SEED, g
    g = torch.Generator()
    SEED = seed


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def POC(train_dataset, valid_dataset, test_dataset, y_train, y_valid, y_test, augmented_dataset,
        model: pl.LightningModule, parms: dict = None, mode=None, task="reg", test=True, weighted=False):
    """
    a. try model according to its parms on data
    b. calculate Spearman correlation
    c. calculate R2

    :param: train_dataset: x_train
    :param: valid_dataset: x_valid
    :param: test_dataset: x_test
    :param: y_train:
    :param: y_valid:
    :param: y_test:
    :model: learning model we want to use, must be a pytorch lightening model
    :param: parms: hyper parameters dictionary
    :param: test: bool which says whether to do test prediction and calculate its corr and R2
    :return: r2_tr, r2_val, c_tr, c_val
    """
    num_workers = 0
    # if torch.cuda.is_available():
    #     num_workers = 32
    # load data according to batches:
    trainloader = data_modul.DataLoader(train_dataset, batch_size=parms["batch_size"], num_workers=num_workers)
    testloader = data_modul.DataLoader(test_dataset, batch_size=parms["batch_size"], num_workers=num_workers)
    validloader = data_modul.DataLoader(valid_dataset, batch_size=parms["batch_size"], num_workers=num_workers)
    if augmented_dataset is not None:
        augmentedloader = data_modul.DataLoader(augmented_dataset, batch_size=parms["batch_size"],
                                                num_workers=num_workers)

    model = model(parms, task=task, mode=mode, weighted=weighted)
    # get_and_apply_next_architecture(model)

    # early stopping when there is no change in val loss for 20 epochs, where no change is defined according
    # to min_delta
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=20, min_delta=0.001, mode="min")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        tt = pl.Trainer(precision=32, max_epochs=1000, callbacks=[early_stop_callback], gpus=1, logger=None,
                        progress_bar_refresh_rate=0, checkpoint_callback=False)
    else:
        tt = pl.Trainer(precision=32, logger=TensorBoardLogger('../CNN_model/cnnruns'), max_epochs=1000,
                        callbacks=[early_stop_callback])

    if augmented_dataset is not None:
        tt.fit(model, [trainloader, augmentedloader], validloader)

    tt.fit(model, trainloader, validloader)

    pred_train = model.predict(trainloader)
    pred_valid = model.predict(validloader)

    if task == 'reg':
        if test:
            pred_test = model.predict(testloader)
            r2 = r2_score(y_test, pred_test)
            c = spearmanr(y_test, pred_test)[0]
            print(f"Test R2: {r2}\n"
                  f"Test Corr: {c}")
        r2_tr = r2_score(y_train, pred_train)
        r2_val = r2_score(y_valid, pred_valid)
        c_tr = spearmanr(y_train, pred_train)[0]
        c_val = spearmanr(y_valid, pred_valid)[0]

        r2_tr_round = r2_score(np.round(y_train), np.round(pred_train))
        r2_val_round = r2_score(np.round(y_valid), np.round(pred_valid))
        return r2_tr, r2_val, c_tr, c_val, r2, c

    labels = range(model.num_of_classes) if model.num_of_classes > 2 else None
    if task == "class":
        if test:
            pred_test = model.predict(testloader)
            acc = accuracy_score(y_test, np.argmax(pred_test, 1) if model.num_of_classes > 2 else pred_test)
            auc = roc_auc_score(y_test, pred_test, multi_class="ovr", labels=labels)
            f1_micro = f1_score(y_test, np.argmax(pred_test, 1) if model.num_of_classes > 2 else pred_test,
                                labels=labels,
                                average="micro")
            f1_macro = f1_score(y_test, np.argmax(pred_test, 1) if model.num_of_classes > 2 else pred_test,
                                labels=labels,
                                average="macro")
            cm = confusion_matrix(y_test, np.argmax(pred_test, 1) if model.num_of_classes > 2 else pred_test,
                                  labels=labels)
            # print(f"Test acc: {acc}\n"
            # f"Test AUC: {auc}")
        acc_tr = accuracy_score(y_train, np.argmax(pred_train, 1) if model.num_of_classes > 2 else pred_train)
        acc_val = accuracy_score(y_valid, np.argmax(pred_valid, 1) if model.num_of_classes > 2 else pred_valid)
        auc_tr = roc_auc_score(y_train, pred_train, multi_class="ovr", labels=labels)
        auc_val = roc_auc_score(y_valid, pred_valid, multi_class="ovr", labels=labels)
        f1_tr_micro = f1_score(y_train, np.argmax(pred_train, 1) if model.num_of_classes > 2 else pred_train,
                               labels=labels,
                               average="micro")
        f1_tr_macro = f1_score(y_train, np.argmax(pred_train, 1) if model.num_of_classes > 2 else pred_train,
                               labels=labels,
                               average="macro")
        f1_val_micro = f1_score(y_valid, np.argmax(pred_valid, 1) if model.num_of_classes > 2 else pred_valid,
                                labels=labels,
                                average="micro")
        f1_val_macro = f1_score(y_valid, np.argmax(pred_valid, 1) if model.num_of_classes > 2 else pred_valid,
                                labels=labels,
                                average="macro")
        return acc_tr, acc_val, auc_tr, auc_val, f1_tr_micro, f1_val_micro, f1_tr_macro, f1_val_macro, acc, auc, f1_micro, f1_macro, cm


def projection(df: pd.DataFrame):
    """
    draws the samples in a 3 dim graph where the samples of each exp are in a different color
    :param df: 3 dim pca from mip mlp
    :return: None
    """
    expers_dim1 = defaultdict(list)
    expers_dim2 = defaultdict(list)
    expers_dim3 = defaultdict(list)
    for i in df.iloc:
        expers_dim1[i.name[:i.name.find('_')]].append(i[0])
        expers_dim2[i.name[:i.name.find('_')]].append(i[1])
        try:
            expers_dim3[i.name[:i.name.find('_')]].append(i[2])
        except:
            pass
    colors = ['r', 'b', 'g', 'purple', 'orange', 'black']
    if len(expers_dim3) == 0:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, k in enumerate(expers_dim1):
            ax.scatter(expers_dim1[k], expers_dim2[k], c=colors[i], label=k)
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i, k in enumerate(expers_dim1):
            ax.scatter(expers_dim1[k], expers_dim2[k], expers_dim3[k], c=
            colors[i], label=k)
    plt.legend()
    plt.show()


def load_data_2d_train_test(otu_train, otu_test, path_of_2D_matrix, tag_train, tag_test, group, biomarkers=None,
                            complex=False):
    """
    a. load 2 dimensional microbiome data days after transplant data and tag data
    according to mapping file
    b. split train and test, while the test is 0.15 of the data
    :param donors: processed OTUS
    :param mapping: mapping file
    :param a_div: tag
    :return: X_train, X_test, y_train, y_test, d_train, d_test
    """

    def load_by_tag(_tag):
        X, y, b = [], [], []
        for index, tag in _tag.iteritems():
            try:
                try:
                    _otu = np.load(f"../{path_of_2D_matrix}/{index}.npy", allow_pickle=True)
                except FileNotFoundError:
                    _otu = np.load(f"{path_of_2D_matrix}/{index}.npy", allow_pickle=True)
                if complex:
                    _otu = fft2(_otu)
                X.append(_otu)
                y.append(tag)
                if biomarkers is not None:
                    bio = biomarkers.loc[index]
                    b.append(bio)
            except KeyError:
                pass
        X = np.array(X)
        y = np.array(y)
        if biomarkers is not None:
            b = np.array(b)
        return X, y, b

    X_train, y_train, b_train = load_by_tag(tag_train)
    X_test, y_test, b_test = load_by_tag(tag_test)

    if len(group) != len(tag_train):
        group = group[tag_train.index]
    # train test split
    if biomarkers is None:
        return X_train, X_test, y_train, y_test, group, [otu_train, tag_train]

    else:
        return X_train, X_test, y_train, y_test, group, otu_train, b_train, b_test


def load_data_3d_train_test(otu_train, otu_test, path_of_2D_matrix, tag_train, tag_test, group, biomarkers=None,
                            complex=False, max_depth=None):
    """
    a. load 2 dimensional microbiome data days after transplant data and tag data
    according to mapping file
    b. split train and test, while the test is 0.15 of the data
    :param donors: processed OTUS
    :param mapping: mapping file
    :param a_div: tag
    :return: X_train, X_test, y_train, y_test, d_train, d_test
    """
    if max_depth is None:
        max_depth = 0

    def load_by_tag(_tag):
        X, y, b = [], [], []
        for subject, tags in _tag.groupby(group):
            subj_X, subj_y, subj_b = [], [], []
            for index, tag in tags.iteritems():
                try:
                    try:
                        _otu = np.load(f"../{path_of_2D_matrix}/{index}.npy", allow_pickle=True)
                    except FileNotFoundError:
                        _otu = np.load(f"{path_of_2D_matrix}/{index}.npy", allow_pickle=True)
                    if complex:
                        _otu = fft2(_otu)
                    subj_X.append(_otu)
                    subj_y.append(tag)
                    if biomarkers is not None:
                        bio = biomarkers.loc[index]
                        subj_b.append(bio)
                except KeyError:
                    pass
            # np.save("../Video/one_person_vid", np.array(subj_X))
            subj_X = np.array(subj_X[-max_depth:])
            subj_y = np.array(subj_y[-max_depth:])
            if biomarkers is not None:
                subj_b = np.array(subj_b)
                b.append(subj_b)
            X.append(subj_X)
            if len(set(subj_y)) != 1:
                Exception("The y values are changing in time, this model does not support this.")
            y.append(subj_y[0])

        X = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in X], batch_first=True)
        y = np.array(y)
        if biomarkers is not None:
            b = np.array(b)
        return X, y, b

    X_train, y_train, b_train = load_by_tag(tag_train)
    X_test, y_test, b_test = load_by_tag(tag_test)

    group = list(range(len(X_train)))
    # train test split
    if biomarkers is None:
        return X_train, X_test, y_train, y_test, group, [otu_train, tag_train]

    else:
        return X_train, X_test, y_train, y_test, group, otu_train, b_train, b_test


def load_data_train_test_valid(X_train, X_test, y_train, y_test, grouping_for_train, org_otu, b_train=None, b_test=None,
                               augment_data=False):
    """
    a. split the train we got from load_data_2d_train_test or from load_data_1d_train_test to
    train and validation without seed. (15% of the whole data as validation)
    b. transform the train, validation and test to tensors
    :param X_train: contains 2D otus
    :param X_test: contains 2D otus (15% of the data)
    :param y_train: a_divs of the train
    :param y_test: a_divs of the test
    :param d_train: days after transplant
    :param d_test: days after transplant
    :return: tensors of train test and validation (x and y)
    """
    augmented_dataset = None
    if b_train is None:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.175, random_state=SEED)  # , random_state=SEED
        sp = [i for i in gss.split(X_train, groups=grouping_for_train)]
        train_idx = sp[0][0]
        valid_idx = sp[0][1]
        try:
            X_train, X_valid = X_train.iloc[train_idx], X_train.iloc[valid_idx]
            y_train, y_valid = y_train.iloc[train_idx], y_train.iloc[valid_idx]
        except AttributeError:
            X_train, X_valid = X_train[train_idx], X_train[valid_idx]
            y_train, y_valid = y_train[train_idx], y_train[valid_idx]

        if augment_data:
            X, y = org_otu
            tag_a, tag_b = seperate_according_to_tag(X.iloc[train_idx], y.iloc[train_idx])

            aug_a = otu22d(augment(tag_a, 3, len(tag_a) * 1), False)
            aug_b = otu22d(augment(tag_b, 3, len(tag_b) * 1), False)

            aug_a = dendogram_ordering(aug_a, tag_a)
            aug_b = dendogram_ordering(aug_b, tag_b)

            a_tag = np.zeros(len(aug_a))
            b_tag = np.ones(len(aug_b))

            aug_X = np.concatenate([aug_a, aug_b])
            aug_y = np.concatenate([a_tag, b_tag])

            augmented_dataset = data_modul.TensorDataset(torch.tensor(aug_X), torch.tensor(aug_y))

        try:
            X_train, X_valid = X_train.to_numpy(), X_valid.to_numpy()
            y_train, y_valid = y_train.to_numpy(), y_valid.to_numpy()
        except:
            pass

        train_dataset = data_modul.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        test_dataset = data_modul.TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
        valid_dataset = data_modul.TensorDataset(torch.tensor(X_valid), torch.tensor(y_valid))

    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.175, random_state=SEED)  # , random_state=SEED
        sp = [i for i in gss.split(X_train, groups=grouping_for_train)]
        train_idx = sp[0][0]
        valid_idx = sp[0][1]
        try:
            X_train, X_valid = X_train.iloc[train_idx].to_numpy(), X_train.iloc[valid_idx].to_numpy()
            y_train, y_valid = y_train.iloc[train_idx].to_numpy(), y_train.iloc[valid_idx].to_numpy()
            b_train, b_valid = b_train.iloc[train_idx].to_numpy(), b_train.iloc[valid_idx].to_numpy()
        except AttributeError:
            X_train, X_valid = X_train[train_idx], X_train[valid_idx]
            y_train, y_valid = y_train[train_idx], y_train[valid_idx]
            b_train, b_valid = b_train[train_idx], b_train[valid_idx]

        train_dataset = data_modul.TensorDataset(torch.tensor(X_train), torch.tensor(y_train), torch.tensor(b_train))
        test_dataset = data_modul.TensorDataset(torch.tensor(X_test), torch.tensor(y_test), torch.tensor(b_test))
        valid_dataset = data_modul.TensorDataset(torch.tensor(X_valid), torch.tensor(y_valid), torch.tensor(b_valid))

    dims = X_train.shape[1:]
    return train_dataset, valid_dataset, test_dataset, y_train, y_valid, y_test, augmented_dataset, dims


# TODO fix
def load_data_1d_train_test(otu_train, otu_test, tag_train, tag_test, mapping, biomarkers=None):
    """
    a. load 1 dimensional microbiome data days after transplant data and tag data
    according to mapping file
    b. split train and test, while the test is 0.15 of the data
    :param donors: processed OTUs
    :param mapping:
    :param a_div: tag
    :return: X_train, X_test, y_train, y_test, d_train, d_test
    """

    if biomarkers is None:
        X_train, X_test = otu_train, otu_test
        y_train, y_test = tag_train, tag_test
        return X_train, X_test.to_numpy(), y_train, y_test.to_numpy(), mapping, None
    else:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        b_train, b_test = b.iloc[train_idx], b.iloc[test_idx]
        return X_train, X_test.to_numpy(), y_train, y_test.to_numpy(), mapping, None, b_train, b_test.to_numpy()


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, choices=["cnn1", "cnn2", "naeive", "3D"], help="Model to use")
    parser.add_argument("data", type=str, help="Name of dataset")
    parser.add_argument("mode", type=str, choices=["1D", "dendogram", "IEEE"])
    parser.add_argument("-tag", "--t", type=str, required=False, default=None)
    parser.add_argument("-biomarkers", "--b", type=bool, required=False, default=False)
    parser.add_argument("-weighted", "--w", type=bool, required=False, default=False)
    parser.add_argument("-augment", "--a", type=bool, required=False, default=False)
    return parser.parse_args()


def main():
    #    sns.heatmap([[10.89285714*0.01 , 7.32142857*0.01, 11.42857143*0.01 , 0.71428571*0.01],
    # [ 4.64285714*0.01,  7.32142857*0.01, 13.21428571*0.01 , 1.60714286*0.01],
    # [ 6.60714286*0.01 , 7.85714286*0.01 ,13.39285714*0.01 , 0.71428571*0.01],
    # [ 1.60714286*0.01 , 3.39285714 *0.01, 6.42857143*0.01,  2.85714286*0.01]], annot=True, fmt=".2%", cmap=plt.get_cmap("Blues"))
    #    plt.show()
    #    exit(0)
    # set_seed(17)
    p = parse()

    # otu, tag, group, biomarkers, path_of_2D_matrix, input_dim, task = load_nni_data("Gastro_vs_oral",
    #                                                                                 D_mode="dendogram")

    print(f"Loading {p.data} with mode {p.mode}")
    otu_train, otu_test, tag_train, tag_test, group, biomarkers, path_of_2D_matrix, input_dim, task = load_nni_data(
        p.data, D_mode=p.mode, tag_name=p.t, after_split=True, with_map=p.b)
    print("Data loaded!")

    D = nni.get_next_parameter()

    print("Splitting to train and test")
    if p.model.lower() == "cnn1":
        model = CNN_1l
        V = load_data_2d_train_test(otu_train, otu_test, path_of_2D_matrix, tag_train, tag_test, group, biomarkers)
        if len(D.values()) == 0:
            D = {

                "l1_loss": 0.3758245982155882,
                "weight_decay": 0.0007067414469528962,
                "lr": 0.001,
                "batch_size": 256,
                "activation": "relu",
                "dropout": 0.05,
                "linear_dim_divider_1": 4,
                "linear_dim_divider_2": 4,
                "kernel_size_a": 7,
                "kernel_size_b": 15,
                "stride": 5,
                "channels": 12
            }

    # edit configuration -> parameters
    if p.model.lower() == "cnn2":
        model = CNN
        # deafult 2 conv CNN parameters, if not in nni mode:
        if len(D.values()) == 0:
            D = {

                 "l1_loss": 0.18035387253504265,
    "weight_decay": 0.07211678723877325,
    "lr": 0.001,
    "batch_size": 256,
    "activation": "tanh",
    "dropout": 0.2,
    "kernel_size_a": 3,
    "kernel_size_b": 6,
    "stride": 1,
    "padding": 3,
    "padding_2": 2,
    "kernel_size_a_2": 2,
    "kernel_size_b_2": 5,
    "stride_2": 2,
    "channels": 4,
    "channels_2": 16,
    "linear_dim_divider_1": 10,
    "linear_dim_divider_2": 11
            }

        V = load_data_2d_train_test(otu_train, otu_test, path_of_2D_matrix, tag_train, tag_test, group, biomarkers)
        # V = load_data_2d_train_test("../2D_otus_our_method/BGU", tag, biomarkers)
    if p.model.lower() == "3d":
        model = CNN_D3
        if len(D.values()) == 0:
            D = {

                "max_depth": 3,
                    "l1_loss": 0.4707487485587963,
                    "weight_decay": 0.026022073518559,
                    "lr": 0.001,
                    "batch_size": 64,
                    "activation": "tanh",
                    "dropout": 0.2,
                    "linear_dim_divider_1": 1,
                    "linear_dim_divider_2": 3,
                    "kernel_size_a": 3,
                    "kernel_size_b": 2,
                    "kernel_size_c": 26,
                    "stride": 3,
                    "channels": 8


            }
        if "max_depth" in D:
            max_depth = D["max_depth"]
        else:
            max_depth = None
        V = load_data_3d_train_test(otu_train, otu_test, path_of_2D_matrix, tag_train, tag_test, group, biomarkers,
                                    max_depth=max_depth)
    if p.model.lower() == "naeive":
        model = Naeive
        if len(D.values()) == 0:
            D = {
                "l1_loss": 0.3996651479346096,
                "weight_decay": 0.09457787118392691,
                "lr": 0.01,
                "batch_size": 64,
                "activation": "tanh",
                "dropout": 0.5,
                "linear_dim_1": 201,
                "linear_dim_2": 136

            }

        V = load_data_1d_train_test(otu_train, otu_test, tag_train, tag_test, group, biomarkers=biomarkers)
    all_r2_tr, all_r2_te, all_c_tr, all_c_te, all_r2_test, all_c_test = [], [], [], [], [], []
    all_f1_mi_tr, all_f1_mi_te, all_f1_mi_test = [], [], []
    all_f1_ma_tr, all_f1_ma_te, all_f1_ma_test = [], [], []
    e = 0
    CM = None
    while e < 10:
        try:
            # e-cross validation (split the train to train and validation) and transform everything to tensor:
            print("splitting to train and validation")
            V1 = load_data_train_test_valid(*V,
                                            augment_data=p.a)  # *V send the parameters of V one by one #,augment_data=True

            bio_dim = 0 if biomarkers is None else biomarkers.shape[1]
            if V1[-1] != input_dim and len(V1[-1]) != 3:
                Warning(
                    f"The input dim of the dataframe {input_dim} and of the loaded data {V1[-1]} isn't the same. The reason is probably"
                    f" bacteria without a full name. Using the loaded data dim {V1[-1]}")
                if type(input_dim) is int:
                    input_dim = V1[-1][0] + bio_dim
                elif len(input_dim) == 2:
                    input_dim = (V1[-1][0], V1[-1][1] + bio_dim)
            if len(V1[-1]) == 3:
                input_dim = tuple(V1[-1])
            V1 = V1[:-1]

            D["input_dim"] = input_dim

            try:
                r2_tr, r2_te, c_tr, c_te, f1_tr_mi, f1_te_mi, f1_tr_ma, f1_te_ma, r2, c, f1_mi, f1_ma, cm = POC(*V1,
                                                                                                                model=model,
                                                                                                                parms=D,
                                                                                                                mode=biomarkers,
                                                                                                                task=task,
                                                                                                                weighted=p.w)
                print(cm)
                if CM is None:
                    CM = cm
                else:
                    CM += cm
            except ValueError as ve:
                if ve.args[0] == "Only one class present in y_true. ROC AUC score is not defined in that case.":
                    e += 0.25
                    continue
                else:
                    raise ve
            e += 1
            # all_r2_tr_round.append(r2_tr_round)
            # all_r2_val_round.append(r2_val_round)
            all_r2_tr.append(r2_tr)
            all_r2_te.append(r2_te)
            all_c_tr.append(c_tr)
            all_c_te.append(c_te)
            all_f1_mi_tr.append(f1_tr_mi)
            all_f1_mi_te.append(f1_te_mi)
            all_f1_ma_tr.append(f1_tr_ma)
            all_f1_ma_te.append(f1_te_ma)
            all_r2_test.append(r2)
            all_c_test.append(c)
            all_f1_mi_test.append(f1_mi)
            all_f1_ma_test.append(f1_ma)
            print(f"R2 train:{r2_tr},\n"
                  f"R2 valid: {r2_te},\n"
                  f"corr train: {c_tr},\n"
                  f"corr valid: {c_te},\n"
                  f"f1 micro train: {f1_tr_mi},\n"
                  f"f1 micro valid: {f1_te_mi},\n"
                  f"f1 macro valid: {f1_tr_ma},\n"
                  f"f1 macro valid: {f1_te_ma},\n"
                  f"R2 test: {r2},\n"
                  f"corr test: {c},\n"
                  f"f1 micro test: {f1_mi},\n"
                  f"f1 macro test: {f1_ma}")
        except Exception as ex:
            raise (ex)
            print(ex)
            nni.report_final_result(-np.inf)

    print(f"\nSTD:")
    print(f"R2 train:{np.std(all_r2_tr)},\n"
          f"R2 valid: {np.std(all_r2_te)},\n"
          f"R2 test: {np.std(all_r2_test)},\n"
          f"corr train: {np.std(all_c_tr)},\n"
          f"corr valid: {np.std(all_c_te)},\n"
          f"corr test: {np.std(all_c_test)}\n"
          f"f1 micro train: {np.std(all_f1_mi_tr)}\n"
          f"f1 micro valid: {np.std(all_f1_mi_te)}\n"
          f"f1 micro test: {np.std(all_f1_mi_test)}\n"
          f"f1 macro train: {np.std(all_f1_ma_tr)}\n"
          f"f1 macro valid: {np.std(all_f1_ma_te)}\n"
          f"f1 macro test: {np.std(all_f1_ma_test)}\n"

          )  # f"R2 rounded train: {np.std(all_r2_tr_round)},\n"
    # f"R2 rounded valid: {np.std(all_r2_val_round)}"
    print(f"Mean:")
    print(f"R2 train:{np.mean(all_r2_tr)},\n"
          f"R2 valid: {np.mean(all_r2_te)},\n"
          f"R2 test: {np.mean(all_r2_test)},\n"
          f"corr train: {np.mean(all_c_tr)},\n"
          f"corr valid: {np.mean(all_c_te)},\n"
          f"corr test: {np.mean(all_c_test)}\n"
          f"f1 micro train: {np.mean(all_f1_mi_tr)}\n"
          f"f1 micro valid: {np.mean(all_f1_mi_te)}\n"
          f"f1 micro test: {np.mean(all_f1_mi_test)}\n"
          f"f1 macro train: {np.mean(all_f1_ma_tr)}\n"
          f"f1 macro valid: {np.mean(all_f1_ma_te)}\n"
          f"f1 macro test: {np.mean(all_f1_ma_test)}\n"

          )  # f"R2 rounded train: {np.mean(all_r2_tr_round)},\n"
    # f"R2 rounded valid: {np.mean(all_r2_val_round)}"

    # report the mean results of e validations

    print("Final cm:")
    print(CM / (e + 1))
    print()
    print((CM / CM.sum()) * 100)
    CM = CM / CM.sum()

    if task == "reg":
        nni.report_final_result(np.mean(all_r2_te))
    elif task == "class":
        nni.report_final_result(np.mean(all_c_te))

    if nni.get_trial_id() == "STANDALONE":
        # Dont run on nni runs
        sns.heatmap(CM, annot=True, fmt=".2%", cmap=plt.get_cmap("Blues"))
        plt.show()

        with open(f"main_nni_tt_results/{SEED}.txt", "w") as f:
            f.write(str(np.mean(all_c_test)))


if __name__ == '__main__':
    main()
