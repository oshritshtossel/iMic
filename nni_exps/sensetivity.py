import pickle
import sys
from copy import deepcopy
from random import random

sys.path.insert(0, "..")

from nni_exps.main_nni_runner_tt import POC, projection, load_data_1d_train_test, load_data_2d_train_test, \
    load_data_3d_train_test, load_data_train_test_valid, parse
from microbiome2matrix import augment, seperate_according_to_tag, otu22d, dendogram_ordering
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from Naieve_model.naeive_model import Naeive
from CNN_model.CNN1convlayer import CNN_1l
from torch.utils import data as data_modul
from nni_data_loader import load_nni_data
from CNN_model.CNN2convlayer import CNN
from CNN_model.CNN_D3 import CNN_D3
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

if __name__ == '__main__':
    p = parse()

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
                "l1_loss": 0.8486967774579114,
                "weight_decay": 0.03888772922470264,
                "lr": 0.001,
                "batch_size": 64,
                "activation": "tanh",
                "dropout": 0.3,
                "linear_dim_divider_1": 2,
                "linear_dim_divider_2": 4,
                "kernel_size_a": 4,
                "kernel_size_b": 4,
                "stride": 8,
                "channels": 3
            }

    # edit configuration -> parameters
    if p.model.lower() == "cnn2":
        model = CNN
        # deafult 2 conv CNN parameters, if not in nni mode:
        if len(D.values()) == 0:
            D = {

                "l1_loss": 0.3975871495700267,
                "weight_decay": 0.008304971371077842,
                "lr": 0.001,
                "batch_size": 64,
                "activation": "tanh",
                "dropout": 0,
                "kernel_size_a": 3,
                "kernel_size_b": 6,
                "stride": 3,
                "padding": 2,
                "padding_2": 0,
                "kernel_size_a_2": 1,
                "kernel_size_b_2": 4,
                "stride_2": 3,
                "channels": 9,
                "channels_2": 13,
                "linear_dim_divider_1": 5,
                "linear_dim_divider_2": 6

            }

        V = load_data_2d_train_test(otu_train, otu_test, path_of_2D_matrix, tag_train, tag_test, group, biomarkers)
        # V = load_data_2d_train_test("../2D_otus_our_method/BGU", tag, biomarkers)
    if p.model.lower() == "3d":
        model = CNN_D3
        if len(D.values()) == 0:
            D = {

                "max_depth": 3,
                "l1_loss": 0.5396879241196503,
                "weight_decay": 0.0007381237179369708,
                "lr": 0.001,
                "batch_size": 32,
                "activation": "elu",
                "dropout": 0.3,
                "linear_dim_divider_1": 8,
                "linear_dim_divider_2": 1,
                "kernel_size_a": 3,
                "kernel_size_b": 1,
                "kernel_size_c": 20,
                "stride": 3,
                "channels": 10

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

                "l1_loss": 0.27651068076804586,
                "weight_decay": 0.09241759955261178,
                "lr": 0.001,
                "batch_size": 128,
                "activation": "elu",
                "dropout": 0.4,
                "linear_dim_1": 268,
                "linear_dim_2": 138
            }

        V = load_data_1d_train_test(otu_train, otu_test, tag_train, tag_test, group, biomarkers=biomarkers)

    all_r2_tr, all_r2_te, all_c_tr, all_c_te, all_r2_test, all_c_test = [], [], [], [], [], []
    all_f1_mi_tr, all_f1_mi_te, all_f1_mi_test = [], [], []
    all_f1_ma_tr, all_f1_ma_te, all_f1_ma_test = [], [], []

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

    All_dict = dict()

    for k in D.keys():
        if k == "activation":
            continue
        for i in [0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3]:
            dd = deepcopy(D)
            if k == "l1_loss" or k == "weight_decay" or k == "lr" or k == "dropout":
                dd[k] = D[k] * i
            elif k == "input_dim":
                continue
            else:
                dd[k] = round(D[k] * i)

            # dd["input_dim"] = input_dim
            r2_tr, r2_te, c_tr, c_te, f1_tr_mi, f1_te_mi, f1_tr_ma, f1_te_ma, r2, c, f1_mi, f1_ma = POC(*V1,
                                                                                                        model=model,
                                                                                                        parms=D,
                                                                                                        mode=biomarkers,
                                                                                                        task=task,
                                                                                                        weighted=p.w)

            # e += 1
            # all_r2_tr_round.append(r2_tr_round)
            # all_r2_val_round.append(r2_val_round)
            All_dict[(i, k)] = c
            print(All_dict)
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
        if task == "reg":
            nni.report_final_result(np.mean(all_r2_te))
        elif task == "class":
            nni.report_final_result(np.mean(all_c_te))

    f = open("sensetivity_results/Nugent/CNN2/sensetivity_results10.pkl", "wb")
    pickle.dump(All_dict, f)
    f.close()
