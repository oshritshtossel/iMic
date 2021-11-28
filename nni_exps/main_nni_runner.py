import sys
import random

import sklearn
from torch.version import cuda

sys.path.insert(0, "..")

from microbiome2matrix import augment, seperate_according_to_tag, otu22d, dendogram_ordering
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from Naieve_model.naeive_model import Naeive
from CNN_model.CNN1convlayer import CNN_1l
from torch.utils import data as data_modul
from nni_data_loader import load_nni_data
from CNN_model.CNN2convlayer import CNN
from collections import defaultdict
from scipy.stats import spearmanr
from numpy.fft import fft2

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import inspect
import torch
import nni
import os

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
pl.seed_everything(SEED, workers=True)
torch.use_deterministic_algorithms(True)
g = torch.Generator()
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def POC(train_dataset, valid_dataset, test_dataset, y_train, y_valid, y_test, augmented_dataset,
        model: pl.LightningModule, parms: dict = None, mode=None, task="reg", test=True):
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

    model = model(parms, task=task, mode=mode)
    # get_and_apply_next_architecture(model)

    # early stopping when there is no change in val loss for 20 epochs, where no change is defined according
    # to min_delta
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=20, min_delta=0.001, mode="min")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        tt = pl.Trainer(precision=32, max_epochs=1000, callbacks=[early_stop_callback], gpus=1, logger=None,
                        progress_bar_refresh_rate=0, checkpoint_callback=False)
    else:
        tt = pl.Trainer(precision=32, logger=TensorBoardLogger('../../CNN_model/cnnruns'), max_epochs=1000,
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
    if task == "class":
        if test:
            pred_test = model.predict(testloader)
            acc = accuracy_score(y_test, pred_test)
            auc = roc_auc_score(y_test, pred_test)
            # print(f"Test acc: {acc}\n"
            # f"Test AUC: {auc}")
        acc_tr = accuracy_score(y_train, pred_train)
        acc_val = accuracy_score(y_valid, pred_valid)
        auc_tr = roc_auc_score(y_train, pred_train)
        auc_val = roc_auc_score(y_valid, pred_valid)
        return acc_tr, acc_val, auc_tr, auc_val, acc, auc


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


def load_data_2d_train_test(otu, path_of_data, tags: pd.Series, mapping, biomarkers=None, complex=False):
    """
    a. load 2 dimensional microbiome data days after transplant data and tag data
    according to mapping file
    b. split train and test, while the test is 0.15 of the data
    :param donors: processed OTUS
    :param mapping: mapping file
    :param a_div: tag
    :return: X_train, X_test, y_train, y_test, d_train, d_test
    """
    indexes = set(tags.index).intersection(otu.index)

    if biomarkers is not None:
        indexes = indexes.intersection(biomarkers.index)
        biomarkers = biomarkers.loc[indexes]
    if pd.Index not in inspect.getmro(type(mapping)):
        indexes = indexes.intersection(mapping.index)
        groups = mapping.loc[indexes]
        groups = np.array(groups.tolist())
    else:
        groups = np.array(list(indexes))
    X = otu.loc[indexes]
    # y = tags.loc[indexes]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)# random_state=SEED
    sp = [i for i in gss.split(X, groups=groups)]
    train_idx = sp[0][0]
    test_idx = sp[0][1]

    # Make sure that test and train are separated
    assert len(set(groups[train_idx]).intersection(groups[test_idx])) == 0

    if pd.Index not in inspect.getmro(type(mapping)):
        grouping_for_train = mapping.iloc[train_idx]
    else:
        grouping_for_train = groups[train_idx]

    _X, y, b = [], [], []

    for index, tag in tags.iteritems():
        try:
            try:
                _otu = np.load(f"../{path_of_data}/{index}.npy", allow_pickle=True)
            except FileNotFoundError:
                _otu = np.load(f"{path_of_data}/{index}.npy", allow_pickle=True)
            if complex:
                _otu = fft2(_otu)
            _X.append(_otu)
            y.append(tag)
            if biomarkers is not None:
                bio = biomarkers.loc[index]
                b.append(bio)
        except KeyError:
            pass
    _X = np.array(_X)
    y = np.array(y)
    if biomarkers is not None:
        b = np.array(b)

    # train test split
    if biomarkers is None:
        X_train, X_test = _X[train_idx], _X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        return X_train, X_test, y_train, y_test, grouping_for_train, [X.iloc[train_idx], tags.iloc[train_idx]]

    else:
        X_train, X_test = _X[train_idx], _X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        b_train, b_test = b[train_idx], b[test_idx]
        return X_train, X_test, y_train, y_test, grouping_for_train, X.iloc[train_idx], b_train, b_test


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
        gss = GroupShuffleSplit(n_splits=1, test_size=0.175, random_state=SEED) #, random_state=SEED
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
            X_train, X_valid = X_train.to_numpy(), X_train.to_numpy()
            y_train, y_valid = y_train.to_numpy(), y_train.to_numpy()
        except:
            pass

        train_dataset = data_modul.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        test_dataset = data_modul.TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
        valid_dataset = data_modul.TensorDataset(torch.tensor(X_valid), torch.tensor(y_valid))

    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.175, random_state=SEED) #, random_state=SEED
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


def load_data_1d_train_test(otu, tags: pd.Series, mapping, biomarkers=None):
    """
    a. load 1 dimensional microbiome data days after transplant data and tag data
    according to mapping file
    b. split train and test, while the test is 0.15 of the data
    :param donors: processed OTUs
    :param mapping:
    :param a_div: tag
    :return: X_train, X_test, y_train, y_test, d_train, d_test
    """
    indexes = set(tags.index).intersection(otu.index)

    if biomarkers is not None:
        indexes = indexes.intersection(biomarkers.index)
        b = biomarkers.loc[indexes]
    if pd.Index not in inspect.getmro(type(mapping)):
        # if type(mapping) is not pd.Index:
        indexes = indexes.intersection(mapping.index)
        groups = mapping.loc[indexes]
        groups = np.array(groups.tolist())
    else:
        groups = np.array(list(indexes))

    X = otu.loc[indexes]
    y = tags.loc[indexes]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED) #, random_state=SEED
    sp = [i for i in gss.split(X, groups=groups)]
    train_idx = sp[0][0]
    test_idx = sp[0][1]

    # Make sure that test and train are separated
    assert len(set(groups[train_idx]).intersection(groups[test_idx])) == 0

    # if type(mapping) is not pd.Index:
    if pd.Index not in inspect.getmro(type(mapping)):
        grouping_for_train = mapping.iloc[train_idx]
    else:
        grouping_for_train = group[train_idx]
    try:
        del X["id"]
    except:
        pass

    if biomarkers is None:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        return X_train, X_test.to_numpy(), y_train, y_test.to_numpy(), grouping_for_train, None
    else:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        b_train, b_test = b.iloc[train_idx], b.iloc[test_idx]
        return X_train, X_test.to_numpy(), y_train, y_test.to_numpy(), grouping_for_train, None, b_train, b_test.to_numpy()


if __name__ == '__main__':
    data = sys.argv[2]
    d_mode = sys.argv[3]
    tag = None
    try:
        tag = sys.argv[4]
    except:
        tag = None
    try:
        with_map = sys.argv[5]
    except:
        with_map = False
    # otu, tag, group, biomarkers, path_of_2D_matrix, input_dim, task = load_nni_data("Gastro_vs_oral",
    #                                                                                 D_mode="dendogram")

    print(f"Loading {data} with mode {d_mode}")
    otu, tag, group, biomarkers, path_of_2D_matrix, input_dim, task = load_nni_data(data, D_mode=d_mode, tag_name=tag,
                                                                                    with_map=with_map)
    print("Data loaded!")

    m = sys.argv[1]
    D = nni.get_next_parameter()

    print("Splitting to train and test")
    if m.lower() == "cnn1":
        model = CNN_1l
        V = load_data_2d_train_test(otu, path_of_2D_matrix, tag, group, biomarkers)
        if len(D.values()) == 0:
            D = {

                    "l1_loss": 0.11482979116939632,
    "weight_decay": 0.016411989557822505,
    "lr": 0.001,
    "batch_size": 32,
    "activation": "elu",
    "dropout": 0.5,
    "linear_dim_divider_1": 2,
    "linear_dim_divider_2": 5,
    "kernel_size_a": 5,
    "kernel_size_b": 15,
    "stride": 3,
    "channels": 7
            }

    # edit configuration -> parameters
    if m.lower() == "cnn2":
        model = CNN
        # deafult 2 conv CNN parameters, if not in nni mode:
        if len(D.values()) == 0:
            D = {
                "l1_loss": 0.0016083011391494102,
                "weight_decay": 0.13817689121075416,
                "lr": 0.001,
                "batch_size": 64,
                "activation": "tanh",
                "dropout": 0.05,
                "kernel_size_a": 2,
                "kernel_size_b": 6,
                "stride": 2,
                "padding": 1,
                "padding_2": 1,
                "kernel_size_a_2": 1,
                "kernel_size_b_2": 5,
                "stride_2": 4,
                "channels": 8,
                "channels_2": 9,
                "linear_dim_divider_1": 10,
                "linear_dim_divider_2": 5
            }

        V = load_data_2d_train_test(otu, path_of_2D_matrix, tag, group, biomarkers)
        # V = load_data_2d_train_test("../2D_otus_our_method/BGU", tag, biomarkers)
    if m.lower() == "naeive":
        model = Naeive
        if len(D.values()) == 0:
            D = {

                    "l1_loss": 0.23026142207312034,
                    "weight_decay": 0.001813273407431773,
                    "lr": 0.001,
                    "batch_size": 32,
                    "activation": "elu",
                    "dropout": 0.5,
                    "linear_dim_1": 280,
                    "linear_dim_2": 80


            }

        V = load_data_1d_train_test(otu, tag, group, biomarkers=biomarkers)
    all_r2_tr, all_r2_te, all_c_tr, all_c_te, all_r2_test, all_c_test = [], [], [], [], [], []
    e = 0
    while e < 10:
        try:
            # e-cross validation (split the train to train and validation) and transform everything to tensor:
            print("splitting to train and validation")
            V1 = load_data_train_test_valid(*V
                                           )  # *V send the parameters of V one by one #,augment_data=True

            bio_dim = 0 if biomarkers is None else biomarkers.shape[1]
            if V1[-1] != input_dim:
                Warning(
                    f"The input dim of the dataframe {input_dim} and of the loaded data {V1[-1]} isn't the same. The reason is probably"
                    f" bacteria without a full name. Using the loaded data dim {V1[-1]}")
                if type(input_dim) is int:
                    input_dim = V1[-1][0] + bio_dim
                elif len(input_dim) == 2:
                    input_dim = (V1[-1][0], V1[-1][1] + bio_dim)
            V1 = V1[:-1]

            D["input_dim"] = input_dim

            try:
                r2_tr, r2_te, c_tr, c_te, r2, c = POC(*V1, model=model, parms=D, mode=biomarkers,
                                                      task=task)

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
            all_r2_test.append(r2)
            all_c_test.append(c)
            print(f"R2 train:{r2_tr},\n"
                  f"R2 valid: {r2_te},\n"
                  f"corr train: {c_tr},\n"
                  f"corr valid: {c_te},\n"
                  f"R2 test: {r2},\n"
                  f"corr test: {c}")
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

          )  # f"R2 rounded train: {np.std(all_r2_tr_round)},\n"
    # f"R2 rounded valid: {np.std(all_r2_val_round)}"
    print(f"Mean:")
    print(f"R2 train:{np.mean(all_r2_tr)},\n"
          f"R2 valid: {np.mean(all_r2_te)},\n"
          f"R2 test: {np.mean(all_r2_test)},\n"
          f"corr train: {np.mean(all_c_tr)},\n"
          f"corr valid: {np.mean(all_c_te)},\n"
          f"corr test: {np.mean(all_c_test)}\n"

          )  # f"R2 rounded train: {np.mean(all_r2_tr_round)},\n"
    # f"R2 rounded valid: {np.mean(all_r2_val_round)}"

    # report the mean results of e validations
    if task == "reg":
        nni.report_final_result(np.mean(all_r2_te))
    elif task == "class":
        nni.report_final_result(np.mean(all_c_te))
