import pandas as pd
import sys
import os
import numpy as np


def after_split_from_pd(otu_path, index_path_train, index_path_test):
    otu_train = pd.read_csv(index_path_train, index_col=0).index
    otu_test = pd.read_csv(index_path_test,
                           index_col=0).index
    otu = pd.read_csv(otu_path, index_col=0)
    otu_train = otu.loc[otu_train]
    otu_test = otu.loc[otu_test]
    return otu_train, otu_test


def load_nni_data(name_of_dataset, D_mode, dendogram=False, tag_name=None, with_map=False, after_split=False):
    """
    load otu table, tag, map and determine the task of each dataset
    :param name_of_dataset: one of :
    "BGU_T0", "BGU", "Allergy_MILK_T0", "GDM_T2_T3", "IBD", "Gastro_vs_oral"
    :param D_mode: one of : "1D", "IEEE", "dendogram"
                   it impacts the order of the otus, if its sub-pca log or relative sum and on the path.

    :param tag_name: its defualt is None, when there is only 1 tag,
                     in "BGU_T0" or "BGU": is one of:
                     "dsc", "ssc", "vat", "li"
                     in "IBD": is one of:
                     "IBD", "CD", "UC"
    :param with_map: can get True or False
                     True - means using the features in mapping file in addition to microbiome
                     False - means using only the microbiome
    :return: otu, tag,map, input_dim, task
    """
    org_path = os.getcwd()

    if "\\" in os.getcwd():
        while os.getcwd().split("\\")[-1] != "CNN_2D_microbiome":
            os.chdir("..")
    else:
        while os.getcwd().split("/")[-1] != "CNN_2D_microbiome":
            os.chdir("..")

    path_of_2D_matrix = None
    biomarkers = None
    group = None
    _map = None
    otu = None

    d = {"Allergy_MILK_T0": {"task": "class",
                             "tag": "Data/Allergy/Milk_no_milk_no_rep_T0_T1_controls/tag.csv",
                             "external_test_tag": {
                                 "train": "train_test_accord_chaim/Allergy_milk/train_valid_tags.csv",
                                 "test": "train_test_accord_chaim/Allergy_milk/test_tags.csv"},
                             "external_otu": {
                                 "train": "train_test_accord_chaim/Allergy_milk/train_valid_micro_sub_pca_log_tax_7_rare_bact_5.csv",
                                 "test": "train_test_accord_chaim/Allergy_milk/test_micro_sub_pca_log_tax_7_rare_bact_5.csv"},
                             "IEEE": {"path": "2D_OTU_IEEE/Allergy_T0_T1_controls",
                                      "otu": "Data/Allergy/Allergy_otu_sum_relative_rare_bact_5_tax_7.csv"},
                             "dendogram": {"path": "2D_otus_dendogram_ordered/Allergy_T0_T1_controls",
                                           "otu": "2D_otus_dendogram_ordered/Allergy_T0_T1_controls/0_fixed_ordered_GBGU_otu_sub_pca_log_tax_7.scv"},
                             "1D": {"otu": "Data/Allergy/Milk_no_milk_no_rep_T0_T1_controls/otus.csv"}},
         "Allergy_all": {"task": "class",
                         "tag": None,
                         "otu": "2D_otus_dendogram_ordered/Allergy_all_samples/0_fixed_ordered_allergy_otu_sub_pca_log_tax_7.csv",
                         "path": "2D_otus_dendogram_ordered/Allergy_all_samples"},
         "Allergy_T0": {"task": "class",
                        "tag": "Data/Allergy/Allergy_vs_no_no_rep_T0_T1_controls/tag.csv",
                        "external_test_tag": {
                            "train": "train_test_accord_chaim/Allergy_no_allergy/train_valid_tags.csv",
                            "test": "train_test_accord_chaim/Allergy_no_allergy/test_tags.csv"},
                        "external_otu": {
                            "train": "train_test_accord_chaim/Allergy_no_allergy/train_valid_micro_sub_pca_log_tax_7_rare_bact_5.csv",
                            "test": "train_test_accord_chaim/Allergy_no_allergy/test_micro_sub_pca_log_tax_7_rare_bact_5.csv"},
                        "IEEE": {"path": "2D_OTU_IEEE/Allergy_T0_T1_controls",
                                 "otu": "Data/Allergy/Allergy_otu_sum_relative_rare_bact_5_tax_7.csv"},
                        "dendogram": {"path": "2D_otus_dendogram_ordered/Allergy_T0_T1_controls",
                                      "otu": "2D_otus_dendogram_ordered/Allergy_T0_T1_controls/0_fixed_ordered_GBGU_otu_sub_pca_log_tax_7.scv"},
                        "1D": {"otu": "Data/Allergy/Milk_no_milk_no_rep_T0_T1_controls/otus.csv"}
                        },
         "New_allergy_success": {"task": "class",
                                 "tag": "Data/New_Allergy/For_learning/sucsees_tag.csv",
                                 "external_test_tag": {
                                     "train": "train_test_accord_chaim/New_allergy_success/train_valid_tags.csv",
                                     "test": "train_test_accord_chaim/New_allergy_success/test_tags.csv"},
                                 "external_otu": {
                                     "train": "train_test_accord_chaim/New_allergy_success/train_valid_micro_sub_pca_log_tax_7_rare_bact_5.csv",
                                     "test": "train_test_accord_chaim/New_allergy_success/test_micro_sub_pca_log_tax_7_rare_bact_5.csv"},
                                 "dendogram": {"path": "2D_otus_dendogram_ordered/New_allergy",
                                               "otu": "2D_otus_dendogram_ordered/New_allergy/0_fixed_ordered_n_allergy_otu_sub_pca_log_tax_7.csv"},
                                 "1D": {"otu": "Data/New_Allergy/For_learning/otus_sub_pca_tax_7_log.csv"}},
         "New_allergy_oitiger": {"task": "class",
                                 "tag": "Data/New_Allergy/For_learning/OITIGER_tag.csv",
                                 "external_test_tag": {
                                     "train": "train_test_accord_chaim/New_allergy_oitiger/train_valid_tags.csv",
                                     "test": "train_test_accord_chaim/New_allergy_oitiger/test_tags.csv"},
                                 "external_otu": {
                                     "train": "train_test_accord_chaim/New_allergy_success/train_valid_micro_sub_pca_log_tax_7_rare_bact_5.csv",
                                     "test": "train_test_accord_chaim/New_allergy_success/test_micro_sub_pca_log_tax_7_rare_bact_5.csv"},
                                 "dendogram": {"path": "2D_otus_dendogram_ordered/New_allergy",
                                               "otu": "2D_otus_dendogram_ordered/New_allergy/0_fixed_ordered_n_allergy_otu_sub_pca_log_tax_7.csv"},
                                 "1D": {"otu": "Data/New_Allergy/For_learning/otus_sub_pca_tax_7_log.csv"}},
         "GDM_T2_T3": {"task": "class",
                       "tag": "Data/GDM/GDM_tag_2_3.csv",
                       "map": "Data/GDM/GDM_map_2_3.csv",
                       "del_from_map": ["Group", "Status"],
                       "IEEE": {"path": "2D_OTU_IEEE/GDM_2_3_tax_6",
                                "otu": "Data/GDM/GDM_tax6_2_3_relative_sum.csv"},
                       "dendogram": {"path": "2D_otus_dendogram_ordered/GDM_2_3_tax_6",
                                     "otu": "2D_otus_dendogram_ordered/GDM_2_3_tax_6/0_fixed_ordered_GDM_otu_sub_pca_log_tax_6.scv"},
                       "1D": {"otu": "Data/GDM/GDM_tax6_2_3_sub_pca_log.csv"}},
         "GDM_Adi_T2_SALIVA": {"task": "class",
                               "tag": "Data/GDM_Adi/Saliva_T2/tag.csv",
                               "map": "Data/GDM_Adi/Saliva_T2/map.csv",
                               "del_from_map": ["Group"],
                               "IEEE": {"path": "2D_OTU_IEEE/GDM_2_3_tax_6",
                                        "otu": "Data/GDM/GDM_tax6_2_3_relative_sum.csv"},
                               "dendogram": {"path": "2D_otus_dendogram_ordered/GDM_Adi/Saliva_T2",
                                             "otu": "2D_otus_dendogram_ordered/GDM_Adi/Saliva_T2/0_fixed_ordered_GDM_otu_sub_pca_log_tax_7.csv"},
                               "1D": {"otu": "Data/GDM_Adi/Saliva_T2/otus_sub_pca_tax_7_log.csv"}},
         "GDM_Adi_T3_SALIVA": {"task": "class",
                               "tag": "Data/GDM_Adi/Saliva_T3/tag.csv",
                               "map": "Data/GDM_Adi/Saliva_T3/map.csv",
                               "del_from_map": ["Group"],
                               "IEEE": {"path": "2D_OTU_IEEE/GDM_2_3_tax_6",
                                        "otu": "Data/GDM/GDM_tax6_2_3_relative_sum.csv"},
                               "dendogram": {"path": "2D_otus_dendogram_ordered/GDM_Adi/Saliva_T3",
                                             "otu": "2D_otus_dendogram_ordered/GDM_Adi/Saliva_T3/0_fixed_ordered_GDM_otu_sub_pca_log_tax_7.csv"},
                               "1D": {"otu": "Data/GDM_Adi/Saliva_T3/otus_sub_pca_tax_7_log.csv"}},
         "GDM_Adi_T2_T3_SALIVA": {"task": "class",
                                  "tag": "Data/GDM_Adi/Saliva_T2_T3/tag.csv",
                                  "map": "Data/GDM_Adi/Saliva_T2_T3/tag.csv",
                                  "del_from_map": ["Group"],
                                  "IEEE": {"path": "2D_OTU_IEEE/GDM_2_3_tax_6",
                                           "otu": "Data/GDM/GDM_tax6_2_3_relative_sum.csv"},
                                  "dendogram": {"path": "2D_otus_dendogram_ordered/GDM_Adi/Saliva_T2_T3",
                                                "otu": "2D_otus_dendogram_ordered/GDM_Adi/Saliva_T2_T3/0_fixed_ordered_GDM_otu_sub_pca_log_tax_7.csv"},
                                  "1D": {"otu": "Data/GDM_Adi/Saliva_T3/otus_sub_pca_tax_7_log.csv"}
                                  }}

    if name_of_dataset == "Allergy_MILK_T0":
        if after_split == False:
            tag = pd.read_csv("Data/Allergy/Milk_no_milk_no_rep_T0_T1_controls/tag.csv", index_col=0)
            group = tag.index
        else:
            tag_train = pd.read_csv("train_test_accord_chaim/Allergy_milk/train_valid_tags.csv", index_col=0)
            tag_test = pd.read_csv("train_test_accord_chaim/Allergy_milk/test_tags.csv", index_col=0)
            group = tag_train.index

        task = "class"

        if D_mode == "IEEE":
            path_of_2D_matrix = "2D_OTU_IEEE/Allergy_T0_T1_controls"
            if after_split == False:
                otu = pd.read_csv("Data/Allergy/Allergy_otu_sum_relative_rare_bact_5_tax_7.csv", index_col=0)
            else:
                otu_train, otu_test = after_split_from_pd(
                    "Data/Allergy/Allergy_otu_sum_relative_rare_bact_5_tax_7.csv",
                    "train_test_accord_chaim/Allergy_milk/train_valid_micro_sub_pca_log_tax_7_rare_bact_5.csv",
                    "train_test_accord_chaim/Allergy_milk/test_micro_sub_pca_log_tax_7_rare_bact_5.csv"
                )


        elif D_mode == "1D":
            if after_split == False:
                otu = pd.read_csv("Data/Allergy/Milk_no_milk_no_rep_T0_T1_controls/otus.csv", index_col=0)
            else:
                otu_train = pd.read_csv(
                    "train_test_accord_chaim/Allergy_milk/train_valid_micro_sub_pca_log_tax_7_rare_bact_5.csv",
                    index_col=0)
                otu_test = pd.read_csv(
                    "train_test_accord_chaim/Allergy_milk/test_micro_sub_pca_log_tax_7_rare_bact_5.csv",
                    index_col=0)

        else:
            path_of_2D_matrix = "2D_otus_dendogram_ordered/Allergy_T0_T1_controls"
            if after_split == False:
                otu = pd.read_csv(
                    "2D_otus_dendogram_ordered/Allergy_T0_T1_controls/0_fixed_ordered_GBGU_otu_sub_pca_log_tax_7.scv",
                    index_col=0)
            else:
                otu_train, otu_test = after_split_from_pd(
                    "2D_otus_dendogram_ordered/Allergy_T0_T1_controls/0_fixed_ordered_GBGU_otu_sub_pca_log_tax_7.scv",
                    "train_test_accord_chaim/Allergy_milk/train_valid_micro_sub_pca_log_tax_7_rare_bact_5.csv",
                    "train_test_accord_chaim/Allergy_milk/test_micro_sub_pca_log_tax_7_rare_bact_5.csv"
                )

    if name_of_dataset == "Allergy_all":
        # only for unsupervised training
        otu = pd.read_csv(
            "2D_otus_dendogram_ordered/Allergy_all_samples/0_fixed_ordered_allergy_otu_sub_pca_log_tax_7.csv",
            index_col=0)
        tag = pd.Series(data=np.zeros_like(otu.index), index=otu.index)

        task = "class"

        path_of_2D_matrix = "2D_otus_dendogram_ordered/Allergy_all_samples"

    if name_of_dataset == "Allergy_T0":
        if after_split == False:
            tag = pd.read_csv("Data/Allergy/Allergy_vs_no_no_rep_T0_T1_controls/tag.csv", index_col=0)
            group = tag.index
        else:
            tag_train = pd.read_csv("train_test_accord_chaim/Allergy_no_allergy/train_valid_tags.csv", index_col=0)
            tag_test = pd.read_csv("train_test_accord_chaim/Allergy_no_allergy/test_tags.csv", index_col=0)
            group = tag_train.index

        task = "class"

        if D_mode == "IEEE":
            path_of_2D_matrix = "2D_OTU_IEEE/Allergy_T0_T1_controls"
            if after_split == False:
                otu = pd.read_csv("Data/Allergy/Allergy_otu_sum_relative_rare_bact_5_tax_7.csv", index_col=0)
            else:
                otu_train, otu_test = after_split_from_pd(
                    "Data/Allergy/Allergy_otu_sum_relative_rare_bact_5_tax_7.csv",
                    "train_test_accord_chaim/Allergy_no_allergy/train_valid_micro_sub_pca_log_tax_7_rare_bact_5.csv",
                    "train_test_accord_chaim/Allergy_no_allergy/test_micro_sub_pca_log_tax_7_rare_bact_5.csv"
                )


        elif D_mode == "1D":
            if after_split == False:
                otu = pd.read_csv("Data/Allergy/Milk_no_milk_no_rep_T0_T1_controls/otus.csv", index_col=0)
            else:
                otu_train = pd.read_csv(
                    "train_test_accord_chaim/Allergy_no_allergy/train_valid_micro_sub_pca_log_tax_7_rare_bact_5.csv",
                    index_col=0)
                otu_test = pd.read_csv(
                    "train_test_accord_chaim/Allergy_no_allergy/test_micro_sub_pca_log_tax_7_rare_bact_5.csv",
                    index_col=0)


        else:
            path_of_2D_matrix = "2D_otus_dendogram_ordered/Allergy_T0_T1_controls"
            if after_split == False:
                otu = pd.read_csv(
                    "2D_otus_dendogram_ordered/Allergy_T0_T1_controls/0_fixed_ordered_GBGU_otu_sub_pca_log_tax_7.scv",
                    index_col=0)
            else:
                otu_train, otu_test = after_split_from_pd(
                    "2D_otus_dendogram_ordered/Allergy_T0_T1_controls/0_fixed_ordered_GBGU_otu_sub_pca_log_tax_7.scv",
                    "train_test_accord_chaim/Allergy_no_allergy/train_valid_micro_sub_pca_log_tax_7_rare_bact_5.csv",
                    "train_test_accord_chaim/Allergy_no_allergy/test_micro_sub_pca_log_tax_7_rare_bact_5.csv"
                )

    if name_of_dataset == "New_allergy_success":
        if after_split == False:
            tag = pd.read_csv("Data/New_Allergy/For_learning/sucsees_tag.csv", index_col=0)
            group = tag.index
        else:
            tag_train = pd.read_csv("train_test_accord_chaim/New_allergy_success/train_valid_tags.csv", index_col=0)
            tag_test = pd.read_csv("train_test_accord_chaim/New_allergy_success/test_tags.csv", index_col=0)
            group = tag_train.index

        task = "class"
        if D_mode == "1D":
            if after_split == False:
                otu = pd.read_csv("Data/New_Allergy/For_learning/otus_sub_pca_tax_7_log.csv", index_col=0)
            else:
                otu_train = pd.read_csv(
                    "train_test_accord_chaim/New_allergy_success/train_valid_micro_sub_pca_log_tax_7_rare_bact_5.csv",
                    index_col=0)
                otu_test = pd.read_csv(
                    "train_test_accord_chaim/New_allergy_success/test_micro_sub_pca_log_tax_7_rare_bact_5.csv",
                    index_col=0)

        else:
            path_of_2D_matrix = "2D_otus_dendogram_ordered/New_allergy"
            if after_split == False:
                otu = pd.read_csv(
                    "2D_otus_dendogram_ordered/New_allergy/0_fixed_ordered_n_allergy_otu_sub_pca_log_tax_7.csv",
                    index_col=0)
            else:
                otu_train, otu_test = after_split_from_pd(
                    "2D_otus_dendogram_ordered/New_allergy/0_fixed_ordered_n_allergy_otu_sub_pca_log_tax_7.csv",
                    "train_test_accord_chaim/New_allergy_success/train_valid_micro_sub_pca_log_tax_7_rare_bact_5.csv",
                    "train_test_accord_chaim/New_allergy_success/test_micro_sub_pca_log_tax_7_rare_bact_5.csv"
                )

    if name_of_dataset == "New_allergy_oitiger":
        if after_split == False:
            tag = pd.read_csv("Data/New_Allergy/For_learning/OITIGER_tag.csv", index_col=0)
            group = tag.index
        else:
            tag_train = pd.read_csv("train_test_accord_chaim/New_allergy_oitiger/train_valid_tags.csv", index_col=0)
            tag_test = pd.read_csv("train_test_accord_chaim/New_allergy_oitiger/test_tags.csv", index_col=0)
            group = tag_train.index

        task = "class"
        if D_mode == "1D":
            if after_split == False:
                otu = pd.read_csv("Data/New_Allergy/For_learning/otus_sub_pca_tax_7_log.csv", index_col=0)
            else:
                otu_train = pd.read_csv(
                    "train_test_accord_chaim/New_allergy_oitiger/train_valid_micro_sub_pca_log_tax_7_rare_bact_5.csv",
                    index_col=0)
                otu_test = pd.read_csv(
                    "train_test_accord_chaim/New_allergy_oitiger/test_micro_sub_pca_log_tax_7_rare_bact_5.csv",
                    index_col=0)

        else:
            path_of_2D_matrix = "2D_otus_dendogram_ordered/New_allergy"
            if after_split == False:
                otu = pd.read_csv(
                    "2D_otus_dendogram_ordered/New_allergy/0_fixed_ordered_n_allergy_otu_sub_pca_log_tax_7.csv",
                    index_col=0)
            else:
                otu_train, otu_test = after_split_from_pd(
                    "2D_otus_dendogram_ordered/New_allergy/0_fixed_ordered_n_allergy_otu_sub_pca_log_tax_7.csv",
                    "train_test_accord_chaim/New_allergy_success/train_valid_micro_sub_pca_log_tax_7_rare_bact_5.csv",
                    "train_test_accord_chaim/New_allergy_success/test_micro_sub_pca_log_tax_7_rare_bact_5.csv"
                )

    if name_of_dataset == "GDM_T2_T3":
        tag = pd.read_csv("Data/GDM/GDM_tag_2_3.csv", index_col=0)
        _map = pd.read_csv("Data/GDM/GDM_map_2_3.csv", index_col=0)
        group = _map["Group"]
        del _map["Group"]
        del _map["Status"]

        task = "class"

        if D_mode == "IEEE":
            otu = pd.read_csv("Data/GDM/GDM_tax6_2_3_relative_sum.csv", index_col=0)
            path_of_2D_matrix = "2D_OTU_IEEE/GDM_2_3_tax_6"
        elif D_mode == "1D":
            otu = pd.read_csv("Data/GDM/GDM_tax6_2_3_sub_pca_log.csv", index_col=0)

        else:
            otu = pd.read_csv("2D_otus_dendogram_ordered/GDM_2_3_tax_6/0_fixed_ordered_GDM_otu_sub_pca_log_tax_6.scv",
                              index_col=0)
            path_of_2D_matrix = "2D_otus_dendogram_ordered/GDM_2_3_tax_6"

    if name_of_dataset == "GDM_Adi_T2_SALIVA":
        tag = pd.read_csv("Data/GDM_Adi/Saliva_T2/tag.csv", index_col=0)
        _map = pd.read_csv("Data/GDM_Adi/Saliva_T2/map.csv", index_col=0)
        group = _map["Group"]
        del _map["Group"]

        task = "class"

        if D_mode == "IEEE":
            ### HAVE TO UPDATE IF WORKS ##########################
            otu = pd.read_csv("Data/GDM/GDM_tax6_2_3_relative_sum.csv", index_col=0)
            path_of_2D_matrix = "2D_OTU_IEEE/GDM_2_3_tax_6"
        elif D_mode == "1D":
            otu = pd.read_csv("Data/GDM_Adi/Saliva_T2/otus_sub_pca_tax_7_log.csv", index_col=0)

        else:
            otu = pd.read_csv(
                "2D_otus_dendogram_ordered/GDM_Adi/Saliva_T2/0_fixed_ordered_GDM_otu_sub_pca_log_tax_7.csv",
                index_col=0)
            path_of_2D_matrix = "2D_otus_dendogram_ordered/GDM_Adi/Saliva_T2"

    if name_of_dataset == "GDM_Adi_T3_SALIVA":
        tag = pd.read_csv("Data/GDM_Adi/Saliva_T3/tag.csv", index_col=0)
        _map = pd.read_csv("Data/GDM_Adi/Saliva_T3/map.csv", index_col=0)
        group = _map["Group"]
        del _map["Group"]

        task = "class"

        if D_mode == "IEEE":
            ### HAVE TO UPDATE IF WORKS ##########################
            otu = pd.read_csv("Data/GDM/GDM_tax6_2_3_relative_sum.csv", index_col=0)
            path_of_2D_matrix = "2D_OTU_IEEE/GDM_2_3_tax_6"
        elif D_mode == "1D":
            otu = pd.read_csv("Data/GDM_Adi/Saliva_T3/otus_sub_pca_tax_7_log.csv", index_col=0)

        else:
            otu = pd.read_csv(
                "2D_otus_dendogram_ordered/GDM_Adi/Saliva_T3/0_fixed_ordered_GDM_otu_sub_pca_log_tax_7.csv",
                index_col=0)
            path_of_2D_matrix = "2D_otus_dendogram_ordered/GDM_Adi/Saliva_T3"

    if name_of_dataset == "GDM_Adi_T2_T3_SALIVA":
        tag = pd.read_csv("Data/GDM_Adi/Saliva_T2_T3/tag.csv", index_col=0)
        _map = pd.read_csv("Data/GDM_Adi/Saliva_T2_T3/map.csv", index_col=0)
        group = _map["Group"]
        del _map["Group"]

        task = "class"

        if D_mode == "IEEE":
            ### HAVE TO UPDATE IF WORKS ##########################
            otu = pd.read_csv("Data/GDM/GDM_tax6_2_3_relative_sum.csv", index_col=0)
            path_of_2D_matrix = "2D_OTU_IEEE/GDM_2_3_tax_6"
        elif D_mode == "1D":
            otu = pd.read_csv("Data/GDM_Adi/Saliva_T2_T3/otus_sub_pca_tax_7_log.csv", index_col=0)

        else:
            otu = pd.read_csv(
                "2D_otus_dendogram_ordered/GDM_Adi/Saliva_T2_T3/0_fixed_ordered_GDM_otu_sub_pca_log_tax_7.csv",
                index_col=0)
            path_of_2D_matrix = "2D_otus_dendogram_ordered/GDM_Adi/Saliva_T2_T3"

    if name_of_dataset == "IBD":
        _map = pd.read_csv("Data/IBD/map.csv", index_col=0)
        group = _map["Group"]
        del _map["CD_or_UC"]
        if D_mode != "1D":
            del _map["Group"]

        task = "class"

        if D_mode == "IEEE":
            path_of_2D_matrix = "2D_OTU_IEEE/IBD"
            if after_split == False:
                otu = pd.read_csv("Data/IBD/only_otus_sum_relative_tax_7_rare_5.csv", index_col=0)
            else:
                otu_train, otu_test = after_split_from_pd("Data/IBD/only_otus_sum_relative_tax_7_rare_5.csv",
                                                          "train_test_accord_chaim/IBD/train_valid_micro_sub_pca_log_tax_7_rare_bact_5.csv",
                                                          "train_test_accord_chaim/IBD/test_micro_sub_pca_log_tax_7_rare_bact_5.csv"
                                                          )

        elif D_mode == "1D":

            if after_split == False:
                otu = pd.read_csv("Data/IBD/only_otus_sub_pca_log_tax_7_rare_5.csv", index_col=0)
            else:
                otu_train = pd.read_csv(
                    "train_test_accord_chaim/IBD/train_valid_micro_sub_pca_log_tax_7_rare_bact_5.csv", index_col=0)
                otu_test = pd.read_csv("train_test_accord_chaim/IBD/test_micro_sub_pca_log_tax_7_rare_bact_5.csv",
                                       index_col=0)
                group = _map["Group"].loc[otu_train.index]
                del _map["Group"]

        else:
            path_of_2D_matrix = "2D_otus_dendogram_ordered/IBD"
            if after_split == False:
                otu = pd.read_csv("2D_otus_dendogram_ordered/IBD/0_fixed_ordered_IBD_otu_sub_pca_log_tax_7.scv",
                                  index_col=0)
            else:
                otu_train, otu_test = after_split_from_pd(
                    "2D_otus_dendogram_ordered/IBD/0_fixed_ordered_IBD_otu_sub_pca_log_tax_7.scv",
                    "train_test_accord_chaim/IBD/train_valid_micro_sub_pca_log_tax_7_rare_bact_5.csv",
                    "train_test_accord_chaim/IBD/test_micro_sub_pca_log_tax_7_rare_bact_5.csv"
                )
        if tag_name == 'IBD':
            if after_split == False:
                tag = pd.read_csv("Data/IBD/tag_IBD_VS_ALL.csv", index_col=0)
            if after_split == True:
                tag_train = pd.read_csv("train_test_accord_chaim/IBD/train_valid_tags.csv", index_col=0)
                tag_test = pd.read_csv("train_test_accord_chaim/IBD/test_tags.csv", index_col=0)


        elif tag_name == "UC":
            tag = pd.read_csv("Data/IBD/tag_UC_VS_ALL.csv", index_col=0)
        elif tag_name == "CD":
            if after_split == False:
                tag = pd.read_csv("Data/IBD/tag_CD_VS_ALL.csv", index_col=0)
            else:
                tag_train = pd.read_csv("train_test_accord_chaim/IBD_CD/train_valid_tags.csv", index_col=0)
                tag_test = pd.read_csv("train_test_accord_chaim/IBD_CD/test_tags.csv", index_col=0)
        elif tag_name == "CD_UC":
            tag = pd.read_csv("Data/IBD/tag_p_id_UC_VS_CD.csv", index_col=0)

    if name_of_dataset == "Cirrhosis":
        tag = pd.read_csv("Data/Cirrhosis_Knight_Lab/tag.csv", index_col=0)

        group = tag.index
        # del map["Group"]
        # del map["Tag"]
        biomarkers = _map
        task = "class"

        if D_mode == "IEEE":
            otu = pd.read_csv(
                "Data/Cirrhosis_Knight_Lab/Cirrhosis_Knight_Lab_otu_sum_relative_rare_bact_5_tax_7_only_after_git_preprocess.csv",
                index_col=0)
            path_of_2D_matrix = "2D_OTU_IEEE/Cirrhosis_Knight_Lab"
        elif D_mode == "1D":
            otu = pd.read_csv(
                "Data/Cirrhosis_Knight_Lab/Cirrhosis_Knight_Lab_otu_sub_pca_log_rare_bact_5_tax_7_only_after_git_preprocess.csv",
                index_col=0)

        else:
            otu = pd.read_csv(
                "2D_otus_dendogram_ordered/Cirrhosis_Knight_Lab/0_fixed_ordered_Cirrhosis_Knight_Lab_otu_sub_pca_log_tax_7.csv",
                index_col=0)
            path_of_2D_matrix = "2D_otus_dendogram_ordered/Cirrhosis_Knight_Lab"

    if name_of_dataset == "Cirrhosis_no_virus":
        if after_split == False:
            tag = pd.read_csv("Data/Cirrhosis_Knight_Lab/tag.csv", index_col=0)
            group = tag.index
        else:
            tag_train = pd.read_csv('train_test_accord_chaim/Cirrhosis/train_valid_tags.csv', index_col=0)
            tag_test = pd.read_csv('train_test_accord_chaim/Cirrhosis/test_tags.csv', index_col=0)
            group = tag_train.index

        # del map["Group"]
        # del map["Tag"]
        biomarkers = _map
        task = "class"

        if D_mode == "IEEE":
            path_of_2D_matrix = "2D_OTU_IEEE/Cirrhosis_Knight_lab_without_viruses"
            if after_split == False:
                otu = pd.read_csv(
                    "Data/Cirrhosis_Knight_Lab/mean_tax_7_relative_rare_bact_5_without_viruses.csv",
                    index_col=0)
            else:
                otu_train, otu_test = after_split_from_pd(
                    "Data/Cirrhosis_Knight_Lab/mean_tax_7_relative_rare_bact_5_without_viruses.csv",
                    "train_test_accord_chaim/Cirrhosis/train_valid_micro_sub_pca_log_tax_7_rare_bact_5.csv",
                    "train_test_accord_chaim/Cirrhosis/test_micro_sub_pca_log_tax_7_rare_bact_5.csv"
                )


        elif D_mode == "1D":
            if after_split == False:
                otu = pd.read_csv("Data/Cirrhosis_Knight_Lab/sub_pca_tax_7_log_rare_bact_5_without_viruses.csv",
                                  index_col=0)
            else:
                otu_train = pd.read_csv(
                    "train_test_accord_chaim/Cirrhosis/train_valid_micro_sub_pca_log_tax_7_rare_bact_5.csv",
                    index_col=0)
                otu_test = pd.read_csv("train_test_accord_chaim/Cirrhosis/test_micro_sub_pca_log_tax_7_rare_bact_5.csv",
                                       index_col=0)


        else:
            path_of_2D_matrix = "2D_otus_dendogram_ordered/Cirrhosis_Knight_lab_without_viruses"
            if after_split == False:
                otu = pd.read_csv(
                    "2D_otus_dendogram_ordered/Cirrhosis_Knight_lab_without_viruses/0_fixed_ordered_Cirrhosis_otu_sub_pca_log_tax_7.csv",
                    index_col=0)

            else:
                otu_train, otu_test = after_split_from_pd(
                    "2D_otus_dendogram_ordered/Cirrhosis_Knight_lab_without_viruses/0_fixed_ordered_Cirrhosis_otu_sub_pca_log_tax_7.csv",
                    "train_test_accord_chaim/Cirrhosis/train_valid_micro_sub_pca_log_tax_7_rare_bact_5.csv",
                    "train_test_accord_chaim/Cirrhosis/test_micro_sub_pca_log_tax_7_rare_bact_5.csv"
                )

    if name_of_dataset == "Male_vs_female":
        if after_split == False:
            tag = pd.read_csv("Data/Male_vs_female/tag.csv", index_col=0)
            group = tag.index
        else:
            tag_train = pd.read_csv("train_test_accord_chaim/Male_vs_female/train_valid_tags.csv", index_col=0)
            tag_test = pd.read_csv("train_test_accord_chaim/Male_vs_female/test_tags.csv", index_col=0)
            group = tag_train.index

        # del map["Group"]
        # del map["Tag"]
        biomarkers = _map
        task = "class"

        if D_mode == "IEEE":
            path_of_2D_matrix = "2D_OTU_IEEE/Male_vs_female"
            if after_split == False:
                otu = pd.read_csv(
                    "Data/Male_vs_female/Male_vs_female_otu_sum_relative_rare_bact_5_tax_7_only_after_git_preprocess.csv",
                    index_col=0)
            else:
                otu_train, otu_test = after_split_from_pd(
                    "Data/Male_vs_female/Male_vs_female_otu_sum_relative_rare_bact_5_tax_7_only_after_git_preprocess.csv",
                    "train_test_accord_chaim/Male_vs_female/train_valid_micro_sub_pca_log_tax_7_rare_bact_5.csv",
                    "train_test_accord_chaim/Male_vs_female/test_micro_sub_pca_log_tax_7_rare_bact_5.csv"
                )


        elif D_mode == "1D":
            if after_split == False:
                otu = pd.read_csv(
                    "Data/Male_vs_female/Male_vs_female_otu_sub_pca_log_rare_bact_5_tax_7_only_after_git_preprocess.csv",
                    index_col=0)

            else:
                otu_train = pd.read_csv(
                    "train_test_accord_chaim/Male_vs_female/train_valid_micro_sub_pca_log_tax_7_rare_bact_5.csv",
                    index_col=0)
                otu_test = pd.read_csv(
                    "train_test_accord_chaim/Male_vs_female/test_micro_sub_pca_log_tax_7_rare_bact_5.csv",
                    index_col=0)


        else:
            path_of_2D_matrix = "2D_otus_dendogram_ordered/Male_vs_female"
            if after_split == False:
                otu = pd.read_csv(
                    "2D_otus_dendogram_ordered/Male_vs_female/0_fixed_ordered_Male_vs_female_otu_sub_pca_log_tax_7.csv",
                    index_col=0)
            else:
                otu_train, otu_test = after_split_from_pd(
                    "2D_otus_dendogram_ordered/Male_vs_female/0_fixed_ordered_Male_vs_female_otu_sub_pca_log_tax_7.csv",
                    "train_test_accord_chaim/Male_vs_female/train_valid_micro_sub_pca_log_tax_7_rare_bact_5.csv",
                    "train_test_accord_chaim/Male_vs_female/test_micro_sub_pca_log_tax_7_rare_bact_5.csv"
                )

    if name_of_dataset.lower() == "white_vs_black_vagina":
        if after_split == False:
            tag = pd.read_csv("Data/White_vs_black_vagina/tag.csv", index_col=0)
            group = tag.index
        else:
            tag_train = pd.read_csv("train_test_accord_chaim/Black_vs_white_vagina/train_valid_tags.csv", index_col=0)
            tag_test = pd.read_csv("train_test_accord_chaim/Black_vs_white_vagina/test_tags.csv", index_col=0)
            group = tag_train.index

        # del map["Group"]
        # del map["Tag"]
        biomarkers = _map
        task = "class"

        if D_mode == "IEEE":
            path_of_2D_matrix = "2D_OTU_IEEE/White_vs_black_vagina"
            if after_split == False:
                otu = pd.read_csv(
                    "Data/White_vs_black_vagina/White_vs_black_vagina_otu_sum_relative_rare_bact_5_tax_7_only_after_git_preprocess.csv",
                    index_col=0)
            else:
                otu_train, otu_test = after_split_from_pd(
                    "Data/White_vs_black_vagina/White_vs_black_vagina_otu_sum_relative_rare_bact_5_tax_7_only_after_git_preprocess.csv",
                    "train_test_accord_chaim/Black_vs_white_vagina/train_valid_micro_sub_pca_log_tax_7_rare_bact_5.csv",
                    "train_test_accord_chaim/Black_vs_white_vagina/test_micro_sub_pca_log_tax_7_rare_bact_5.csv"
                )



        elif D_mode == "1D":
            if after_split == False:
                otu = pd.read_csv(
                    "Data/White_vs_black_vagina/White_vs_black_vagina_otu_sub_pca_log_rare_bact_5_tax_7_only_after_git_preprocess.csv",
                    index_col=0)
            else:
                otu_train = pd.read_csv(
                    "train_test_accord_chaim/Black_vs_white_vagina/train_valid_micro_sub_pca_log_tax_7_rare_bact_5.csv",
                    index_col=0)
                otu_test = pd.read_csv(
                    "train_test_accord_chaim/Black_vs_white_vagina/test_micro_sub_pca_log_tax_7_rare_bact_5.csv",
                    index_col=0)


        else:
            path_of_2D_matrix = "2D_otus_dendogram_ordered/White_vs_black_vagina"
            if after_split == False:
                otu = pd.read_csv(
                    "2D_otus_dendogram_ordered/White_vs_black_vagina/0_fixed_ordered_White_vs_black_vagina_otu_sub_pca_log_tax_7.csv",
                    index_col=0)
            else:
                otu_train, otu_test = after_split_from_pd(
                    "2D_otus_dendogram_ordered/White_vs_black_vagina/0_fixed_ordered_White_vs_black_vagina_otu_sub_pca_log_tax_7.csv",
                    "train_test_accord_chaim/Black_vs_white_vagina/train_valid_micro_sub_pca_log_tax_7_rare_bact_5.csv",
                    "train_test_accord_chaim/Black_vs_white_vagina/test_micro_sub_pca_log_tax_7_rare_bact_5.csv"
                )

    if name_of_dataset == "new_allergy_milk":
        if after_split == False:
            tag = pd.read_csv("Data/Allergy/New_allergy_milk/milk_tag.csv", index_col=0)
            group = tag.index
        else:
            tag_train = pd.read_csv("train_test_accord_chaim/New_allergy_milk/train_val_set_milk_tags.csv", index_col=0)
            tag_test = pd.read_csv("train_test_accord_chaim/New_allergy_milk/test_set_milk_tags.csv", index_col=0)
            group = tag_train.index

        # del map["Group"]
        # del map["Tag"]
        biomarkers = _map
        task = "class"

        if D_mode == "IEEE":
            path_of_2D_matrix = "2D_OTU_IEEE/Allergy_T0_T1_controls"
            if after_split == False:
                otu = pd.read_csv(
                    "Data/Allergy/Milk_no_milk_no_rep_T0_T1_controls/otus_relative_sum.csv",
                    index_col=0)
            else:
                otu_train, otu_test = after_split_from_pd(
                    "Data/Allergy/Milk_no_milk_no_rep_T0_T1_controls/otus_relative_sum.csv",
                    "train_test_accord_chaim/New_allergy_milk/train_val_set_milk_microbiome.csv",
                    "train_test_accord_chaim/New_allergy_milk/test_set_milk_microbiome.csv"
                )



        elif D_mode == "1D":
            if after_split == False:
                otu = pd.read_csv(
                    "Data/Allergy/Milk_no_milk_no_rep_T0_T1_controls/otus.csv",
                    index_col=0)
            else:
                otu_train = pd.read_csv(
                    "train_test_accord_chaim/New_allergy_milk/train_val_set_milk_microbiome.csv",
                    index_col=0)
                otu_test = pd.read_csv(
                    "train_test_accord_chaim/New_allergy_milk/test_set_milk_microbiome.csv",
                    index_col=0)


        else:
            path_of_2D_matrix = "2D_otus_dendogram_ordered/Allergy_T0_T1_controls"
            if after_split == False:
                otu = pd.read_csv(
                    "2D_otus_dendogram_ordered/Allergy_T0_T1_controls/0_fixed_ordered_GBGU_otu_sub_pca_log_tax_7.scv",
                    index_col=0)
            else:
                otu_train, otu_test = after_split_from_pd(
                    "2D_otus_dendogram_ordered/Allergy_T0_T1_controls/0_fixed_ordered_GBGU_otu_sub_pca_log_tax_7.scv",
                    "train_test_accord_chaim/New_allergy_milk/train_val_set_milk_microbiome.csv",
                    "train_test_accord_chaim/New_allergy_milk/test_set_milk_microbiome.csv"
                )


    if name_of_dataset == "new_allergy_peanuts":
        if after_split == False:
            tag = pd.read_csv("Data/Allergy/New_allergy_Peanuts/nuts_tag.csv", index_col=0)
            group = tag.index
        else:
            tag_train = pd.read_csv("train_test_accord_chaim/New_allergy_peanut/train_val_set_peanut_tags.csv", index_col=0)
            tag_test = pd.read_csv("train_test_accord_chaim/New_allergy_peanut/test_set_peanut_tags.csv", index_col=0)
            group = tag_train.index

        # del map["Group"]
        # del map["Tag"]
        biomarkers = _map
        task = "class"

        if D_mode == "IEEE":
            path_of_2D_matrix = "2D_OTU_IEEE/Allergy_T0_T1_controls"
            if after_split == False:
                otu = pd.read_csv(
                    "Data/Allergy/Milk_no_milk_no_rep_T0_T1_controls/otus_relative_sum.csv",
                    index_col=0)
            else:
                otu_train, otu_test = after_split_from_pd(
                    "Data/Allergy/Milk_no_milk_no_rep_T0_T1_controls/otus_relative_sum.csv",
                    "train_test_accord_chaim/New_allergy_peanut/train_val_set_peanut_microbiome.csv",
                    "train_test_accord_chaim/New_allergy_peanut/test_set_peanut_microbiome.csv"
                )



        elif D_mode == "1D":
            if after_split == False:
                otu = pd.read_csv(
                    "Data/Allergy/Milk_no_milk_no_rep_T0_T1_controls/otus.csv",
                    index_col=0)
            else:
                otu_train = pd.read_csv(
                    "train_test_accord_chaim/New_allergy_peanut/train_val_set_peanut_microbiome.csv",
                    index_col=0)
                otu_test = pd.read_csv(
                    "train_test_accord_chaim/New_allergy_peanut/test_set_peanut_microbiome.csv",
                    index_col=0)


        else:
            path_of_2D_matrix = "2D_otus_dendogram_ordered/Allergy_T0_T1_controls"
            if after_split == False:
                otu = pd.read_csv(
                    "2D_otus_dendogram_ordered/Allergy_T0_T1_controls/0_fixed_ordered_GBGU_otu_sub_pca_log_tax_7.scv",
                    index_col=0)
            else:
                otu_train, otu_test = after_split_from_pd(
                    "2D_otus_dendogram_ordered/Allergy_T0_T1_controls/0_fixed_ordered_GBGU_otu_sub_pca_log_tax_7.scv",
                    "train_test_accord_chaim//New_allergy_peanut/train_val_set_peanut_microbiome.csv",
                    "train_test_accord_chaim//New_allergy_peanut/test_set_peanut_microbiome.csv"
                )



    if name_of_dataset == "new_allergy_nut":
        if after_split == False:
            tag = pd.read_csv("Data/Allergy/New_allergy_nuts/nuts_tag.csv", index_col=0)
            group = tag.index
        else:
            tag_train = pd.read_csv("train_test_accord_chaim/New_allergy_nut/train_val_set_nut_tags.csv", index_col=0)
            tag_test = pd.read_csv("train_test_accord_chaim/New_allergy_nut/test_set_nut_tags.csv", index_col=0)
            group = tag_train.index

        # del map["Group"]
        # del map["Tag"]
        biomarkers = _map
        task = "class"

        if D_mode == "IEEE":
            path_of_2D_matrix = "2D_OTU_IEEE/Allergy_T0_T1_controls"
            if after_split == False:
                otu = pd.read_csv(
                    "Data/Allergy/Milk_no_milk_no_rep_T0_T1_controls/otus_relative_sum.csv",
                    index_col=0)
            else:
                otu_train, otu_test = after_split_from_pd(
                    "Data/Allergy/Milk_no_milk_no_rep_T0_T1_controls/otus_relative_sum.csv",
                    "train_test_accord_chaim/New_allergy_nut/train_val_set_nut_microbiome.csv",
                    "train_test_accord_chaim/New_allergy_nut/test_set_nut_microbiome.csv"
                )



        elif D_mode == "1D":
            if after_split == False:
                otu = pd.read_csv(
                    "Data/Allergy/Milk_no_milk_no_rep_T0_T1_controls/otus.csv",
                    index_col=0)
            else:
                otu_train = pd.read_csv(
                    "train_test_accord_chaim/New_allergy_nut/train_val_set_nut_microbiome.csv",
                    index_col=0)
                otu_test = pd.read_csv(
                    "train_test_accord_chaim/New_allergy_nut/test_set_nut_microbiome.csv",
                    index_col=0)


        else:
            path_of_2D_matrix = "2D_otus_dendogram_ordered/Allergy_T0_T1_controls"
            if after_split == False:
                otu = pd.read_csv(
                    "2D_otus_dendogram_ordered/Allergy_T0_T1_controls/0_fixed_ordered_GBGU_otu_sub_pca_log_tax_7.scv",
                    index_col=0)
            else:
                otu_train, otu_test = after_split_from_pd(
                    "2D_otus_dendogram_ordered/Allergy_T0_T1_controls/0_fixed_ordered_GBGU_otu_sub_pca_log_tax_7.scv",
                    "train_test_accord_chaim/New_allergy_nut/train_val_set_nut_microbiome.csv",
                    "train_test_accord_chaim/New_allergy_nut/test_set_nut_microbiome.csv"
                )


    if name_of_dataset == "nugent":
        if after_split == False:
            tag = pd.read_csv("Data/Nugent_vagina/tag.csv", index_col=0)
            group = tag.index
        else:
            tag_train = pd.read_csv("train_test_accord_chaim/Nugent_vagina/train_valid_tags.csv", index_col=0)
            tag_test = pd.read_csv("train_test_accord_chaim/Nugent_vagina/tag_test.csv", index_col=0)
            group = tag_train.index

        # del map["Group"]
        # del map["Tag"]
        biomarkers = _map
        task = "class"

        if D_mode == "IEEE":
            path_of_2D_matrix = "2D_OTU_IEEE/Nugent_vagina"
            if after_split == False:
                otu = pd.read_csv(
                    "Data/Nugent_vagina/relative_mean_rare_bact5.csv",
                    index_col=0)
            else:
                otu_train, otu_test = after_split_from_pd(
                    "Data/Nugent_vagina/relative_mean_rare_bact5.csv",
                    "train_test_accord_chaim/Nugent_vagina/train_valid_micro_sub_pca_log_tax_7_rare_bact_5.csv",
                    "train_test_accord_chaim/Nugent_vagina/micro_test.csv"
                )



        elif D_mode == "1D":
            if after_split == False:
                otu = pd.read_csv(
                    "Data/Nugent_vagina/subpca_log_normalization_rare_bact5.csv",
                    index_col=0)
            else:
                otu_train = pd.read_csv(
                    "train_test_accord_chaim/Nugent_vagina/train_valid_micro_sub_pca_log_tax_7_rare_bact_5.csv",
                    index_col=0)
                otu_test = pd.read_csv(
                    "train_test_accord_chaim/Nugent_vagina/micro_test.csv",
                    index_col=0)


        else:
            path_of_2D_matrix = "2D_otus_dendogram_ordered/Nugent_vagina"
            if after_split == False:
                otu = pd.read_csv(
                    "2D_otus_dendogram_ordered/Nugent_vagina/0_fixed_ordered_n_nugent_otu_sub_pca_log_tax_7.csv",
                    index_col=0)
            else:
                otu_train, otu_test = after_split_from_pd(
                    "2D_otus_dendogram_ordered/Nugent_vagina/0_fixed_ordered_n_nugent_otu_sub_pca_log_tax_7.csv",
                    "train_test_accord_chaim/Nugent_vagina/train_valid_micro_sub_pca_log_tax_7_rare_bact_5.csv",
                    "train_test_accord_chaim/Nugent_vagina/micro_test.csv"
                )


    if name_of_dataset == "Diabimmune":
        _map = pd.read_csv("Data_TS/Diabimmune/Milk/mapping.csv", index_col=0)
        group = _map["Group"]
        del _map["age_at_collection"]
        if D_mode != "1D":
            del _map["Group"]

        task = "class"

        if D_mode == "IEEE":  ############## now cant run ########################
            path_of_2D_matrix = "2D_OTU_IEEE/Diabimmune"
            if after_split == False:
                otu = pd.read_csv("Data_TS/Diabimmune/Milk/only_otus_sum_relative_tax_7_rare_5.csv",
                                  index_col=0)  # to change!!!!!!!!!!!!!!!
            else:
                if tag == "milk":
                    otu_train, otu_test = after_split_from_pd(
                        "Data_TS/Diabimmune/Milk/only_otus_sum_relative_tax_7_rare_5.csv",
                        "train_test_accord_chaim/Diabimmune/Milk/train_val_set_Milk_microbiome.csv",
                        "train_test_accord_chaim/Diabimmune/Milk/test_set_Milk_microbiome.csv"
                    )
                if tag == "egg":
                    otu_train, otu_test = after_split_from_pd(
                        "Data_TS/Diabimmune/Milk/only_otus_sum_relative_tax_7_rare_5.csv",
                        "train_test_accord_chaim/Diabimmune/Egg/train_val_set_Egg_microbiome.csv",
                        "train_test_accord_chaim/Diabimmune/Egg/test_set_Egg_microbiome.csv"
                    )
                if tag == "peanuts":
                    otu_train, otu_test = after_split_from_pd(
                        "Data_TS/Diabimmune/Milk/only_otus_sum_relative_tax_7_rare_5.csv",
                        "train_test_accord_chaim/Diabimmune/Peanuts/train_val_set_Peanuts_microbiome.csv",
                        "train_test_accord_chaim/Diabimmune/Peanuts/test_set_Peanuts_microbiome.csv"
                    )
                if tag == "all":
                    otu_train, otu_test = after_split_from_pd(
                        "Data_TS/Diabimmune/Milk/only_otus_sum_relative_tax_7_rare_5.csv",
                        "train_test_accord_chaim/Diabimmune/All/train_val_set_All_microbiome.csv",
                        "train_test_accord_chaim/Diabimmune/All/test_set_All_microbiome.csv"
                    )


        elif D_mode == "1D":

            if after_split == False:
                otu = pd.read_csv("Data_TS/Diabimmune/Milk/subpca_tax7_otus.csv", index_col=0)
            else:
                if tag_name == "milk":
                    otu_train = pd.read_csv(
                        "train_test_accord_chaim/Diabimmune/Milk/train_val_set_Milk_microbiome.csv", index_col=0)
                    otu_test = pd.read_csv("train_test_accord_chaim/Diabimmune/Milk/test_set_Milk_microbiome.csv",
                                           index_col=0)
                    group = _map["Group"].loc[otu_train.index]
                    del _map["Group"]

                if tag_name == "egg":
                    otu_train = pd.read_csv(
                        "train_test_accord_chaim/Diabimmune/Egg/train_val_set_Egg_microbiome.csv",
                        index_col=0)
                    otu_test = pd.read_csv(
                        "train_test_accord_chaim/Diabimmune/Egg/test_set_Egg_microbiome.csv",
                        index_col=0)
                    group = _map["Group"].loc[otu_train.index]
                    del _map["Group"]

                if tag_name == "peanuts":
                    otu_train = pd.read_csv(
                        "train_test_accord_chaim/Diabimmune/Peanuts/train_val_set_Peanuts_microbiome.csv",
                        index_col=0)
                    otu_test = pd.read_csv(
                        "train_test_accord_chaim/Diabimmune/Peanuts/test_set_Peanuts_microbiome.csv",
                        index_col=0)
                    group = _map["Group"].loc[otu_train.index]
                    del _map["Group"]

                if tag_name == "all":
                    otu_train = pd.read_csv(
                        "train_test_accord_chaim/Diabimmune/All/train_val_set_All_microbiome.csv",
                        index_col=0)
                    otu_test = pd.read_csv(
                        "train_test_accord_chaim/Diabimmune/All/test_set_All_microbiome.csv",
                        index_col=0)
                    group = _map["Group"].loc[otu_train.index]
                    del _map["Group"]



        else:
            path_of_2D_matrix = "2D_otus_dendogram_ordered/Diabimmune"
            if after_split == False:
                otu = pd.read_csv(
                    "2D_otus_dendogram_ordered/Diabimmune/0_fixed_ordered_n_diab_otu_sub_pca_log_tax_7.csv",
                    index_col=0)
            else:
                if tag_name == "milk":
                    otu_train, otu_test = after_split_from_pd(
                        "2D_otus_dendogram_ordered/Diabimmune/0_fixed_ordered_n_diab_otu_sub_pca_log_tax_7.csv",
                        "train_test_accord_chaim/Diabimmune/Milk/train_val_set_Milk_microbiome.csv",
                        "train_test_accord_chaim/Diabimmune/Milk/test_set_Milk_microbiome.csv"
                    )
                if tag_name == "egg":
                    otu_train, otu_test = after_split_from_pd(
                        "2D_otus_dendogram_ordered/Diabimmune/0_fixed_ordered_n_diab_otu_sub_pca_log_tax_7.csv",
                        "train_test_accord_chaim/Diabimmune/Egg/train_val_set_Egg_microbiome.csv",
                        "train_test_accord_chaim/Diabimmune/Egg/test_set_Egg_microbiome.csv"
                    )
                if tag_name == "peanuts":
                    otu_train, otu_test = after_split_from_pd(
                        "2D_otus_dendogram_ordered/Diabimmune/0_fixed_ordered_n_diab_otu_sub_pca_log_tax_7.csv",
                        "train_test_accord_chaim/Diabimmune/Peanuts/train_val_set_Peanuts_microbiome.csv",
                        "train_test_accord_chaim/Diabimmune/Peanuts/test_set_Peanuts_microbiome.csv"
                    )
                if tag_name == "all":
                    otu_train, otu_test = after_split_from_pd(
                        "2D_otus_dendogram_ordered/Diabimmune/0_fixed_ordered_n_diab_otu_sub_pca_log_tax_7.csv",
                        "train_test_accord_chaim/Diabimmune/All/train_val_set_All_microbiome.csv",
                        "train_test_accord_chaim/Diabimmune/All/test_set_All_microbiome.csv"
                    )
        if tag_name == 'milk':
            if after_split == False:
                tag = pd.read_csv("Data_TS/Diabimmune/Milk/tag.csv", index_col=0)
            if after_split == True:
                tag_train = pd.read_csv("train_test_accord_chaim/Diabimmune/Milk/train_val_set_Milk_tags.csv",
                                        index_col=0)
                tag_test = pd.read_csv("train_test_accord_chaim/Diabimmune/Milk/test_set_Milk_tags.csv", index_col=0)

        if tag_name == 'egg':
            if after_split == False:
                tag = pd.read_csv("Data_TS/Diabimmune/Egg/tag.csv", index_col=0)
            if after_split == True:
                tag_train = pd.read_csv("train_test_accord_chaim/Diabimmune/Egg/train_val_set_Egg_tags.csv",
                                        index_col=0)
                tag_test = pd.read_csv("train_test_accord_chaim/Diabimmune/Egg/test_set_Egg_tags.csv", index_col=0)

        if tag_name == 'peanuts':
            if after_split == False:
                tag = pd.read_csv("Data_TS/Diabimmune/Peanuts/tag.csv", index_col=0)
            if after_split == True:
                tag_train = pd.read_csv("train_test_accord_chaim/Diabimmune/Peanuts/train_val_set_Peanuts_tags.csv",
                                        index_col=0)
                tag_test = pd.read_csv("train_test_accord_chaim/Diabimmune/Peanuts/test_set_Peanuts_tags.csv",
                                       index_col=0)

        if tag_name == 'all':
            if after_split == False:
                tag = pd.read_csv("Data_TS/Diabimmune/All/tag.csv", index_col=0)
            if after_split == True:
                tag_train = pd.read_csv("train_test_accord_chaim/Diabimmune/All/train_val_set_All_tags.csv",
                                        index_col=0)
                tag_test = pd.read_csv("train_test_accord_chaim/Diabimmune/All/test_set_All_tags.csv", index_col=0)

     #############################################
    # TO CHANGE TO NEW BINARY ALLERGY
    #################################################
    if name_of_dataset == "b_allergy":
        _map = pd.read_csv("Data/Allergy/Allergy_all_times_all_kinds/mapping.csv", index_col=0)
        group = _map["Group"]
        del _map["TIME"]
        if D_mode != "1D":
            del _map["Group"]

        task = "class"

        if D_mode == "IEEE":  ############## now cant run ########################
            path_of_2D_matrix = "2D_OTU_IEEE/Diabimmune"
            if after_split == False:
                otu = pd.read_csv("Data_TS/Diabimmune/Milk/only_otus_sum_relative_tax_7_rare_5.csv",
                                  index_col=0)  # to change!!!!!!!!!!!!!!!
            else:
                if tag == "milk":
                    otu_train, otu_test = after_split_from_pd(
                        "Data/Allergy/Allergy_all_times_all_kinds/otus.csv",
                        "train_test_accord_chaim/binary_allergy_ts/milk/train_val_set_ALL_ALLERGY_microbiome.csv",
                        "train_test_accord_chaim/binary_allergy_ts/milk/test_set_ALL_ALLERGY_microbiome.csv"
                    )
                if tag == "nuts":
                    otu_train, otu_test = after_split_from_pd(
                        "Data/Diabimmune/Milk/only_otus_sum_relative_tax_7_rare_5.csv",
                        "train_test_accord_chaim/Diabimmune/Egg/train_val_set_Egg_microbiome.csv",
                        "train_test_accord_chaim/Diabimmune/Egg/test_set_Egg_microbiome.csv"
                    )
                if tag == "peanuts":
                    otu_train, otu_test = after_split_from_pd(
                        "Data_TS/Diabimmune/Milk/only_otus_sum_relative_tax_7_rare_5.csv",
                        "train_test_accord_chaim/Diabimmune/Peanuts/train_val_set_Peanuts_microbiome.csv",
                        "train_test_accord_chaim/Diabimmune/Peanuts/test_set_Peanuts_microbiome.csv"
                    )
                if tag == "all":
                    otu_train, otu_test = after_split_from_pd(
                        "Data_TS/Diabimmune/Milk/only_otus_sum_relative_tax_7_rare_5.csv",
                        "train_test_accord_chaim/Diabimmune/All/train_val_set_All_microbiome.csv",
                        "train_test_accord_chaim/Diabimmune/All/test_set_All_microbiome.csv"
                    )


        elif D_mode == "1D":

            if after_split == False:
                otu = pd.read_csv("Data/Allergy/Allergy_all_times_all_kinds/otus.csv", index_col=0)
            else:
                if tag_name == "milk":
                    otu_train = pd.read_csv(
                        "train_test_accord_chaim/binary_allergy_ts/milk/train_val_set_ALL_ALLERGY_microbiome.csv", index_col=0)
                    otu_test = pd.read_csv("train_test_accord_chaim/binary_allergy_ts/milk/test_set_ALL_ALLERGY_microbiome.csv",
                                           index_col=0)
                    group = _map["Group"].loc[otu_train.index]
                    del _map["Group"]

                if tag_name == "nuts":
                    otu_train = pd.read_csv(
                        "train_test_accord_chaim/binary_allergy_ts/nuts/train_val_set_ALL_ALLERGY_microbiome.csv",
                        index_col=0)
                    otu_test = pd.read_csv(
                        "train_test_accord_chaim/binary_allergy_ts/nuts/test_set_ALL_ALLERGY_microbiome.csv",
                        index_col=0)
                    group = _map["Group"].loc[otu_train.index]
                    del _map["Group"]

                if tag_name == "peanuts":
                    otu_train = pd.read_csv(
                        "train_test_accord_chaim/binary_allergy_ts/peanuts/train_val_set_ALL_ALLERGY_microbiome.csv",
                        index_col=0)
                    otu_test = pd.read_csv(
                        "train_test_accord_chaim/binary_allergy_ts/peanuts/test_set_ALL_ALLERGY_microbiome.csv",
                        index_col=0)
                    group = _map["Group"].loc[otu_train.index]
                    del _map["Group"]

                if tag_name == "sesame":
                    otu_train = pd.read_csv(
                        "train_test_accord_chaim/binary_allergy_ts/sesame/train_val_set_ALL_ALLERGY_microbiome.csv",
                        index_col=0)
                    otu_test = pd.read_csv(
                        "train_test_accord_chaim/binary_allergy_ts/sesame/test_set_ALL_ALLERGYl_microbiome.csv",
                        index_col=0)
                    group = _map["Group"].loc[otu_train.index]
                    del _map["Group"]



        else:
            path_of_2D_matrix = "2D_otus_dendogram_ordered/Allergy_all_kind_all_times"
            if after_split == False:
                otu = pd.read_csv(
                    "2D_otus_dendogram_ordered/Allergy_all_kind_all_times/0_fixed_ordered_n_all_otu_sub_pca_log_tax_7.csv",
                    index_col=0)
            else:
                if tag_name == "milk":
                    otu_train, otu_test = after_split_from_pd(
                        "2D_otus_dendogram_ordered/Allergy_all_kind_all_times/0_fixed_ordered_n_all_otu_sub_pca_log_tax_7.csv",
                        "train_test_accord_chaim/binary_allergy_ts/milk/train_val_set_ALL_ALLERGY_microbiome.csv",
                        "train_test_accord_chaim/binary_allergy_ts/milk/test_set_ALL_ALLERGY_microbiome.csv"
                    )
                if tag_name == "nuts":
                    otu_train, otu_test = after_split_from_pd(
                        "2D_otus_dendogram_ordered/Allergy_all_kind_all_times/0_fixed_ordered_n_all_otu_sub_pca_log_tax_7.csv",
                        "train_test_accord_chaim/binary_allergy_ts/nuts/train_val_set_ALL_ALLERGY_microbiome.csv",
                        "train_test_accord_chaim/binary_allergy_ts/nuts/test_set_ALL_ALLERGY_microbiome.csv"
                    )
                if tag_name == "peanuts":
                    otu_train, otu_test = after_split_from_pd(
                        "2D_otus_dendogram_ordered/Allergy_all_kind_all_times/0_fixed_ordered_n_all_otu_sub_pca_log_tax_7.csv",
                        "train_test_accord_chaim/binary_allergy_ts/peanuts/train_val_set_ALL_ALLERGY_microbiome.csv",
                        "train_test_accord_chaim/binary_allergy_ts/peanuts/test_set_ALL_ALLERGY_microbiome.csv"
                    )
                if tag_name == "sesame":
                    otu_train, otu_test = after_split_from_pd(
                        "2D_otus_dendogram_ordered/Allergy_all_kind_all_times/0_fixed_ordered_n_all_otu_sub_pca_log_tax_7.csv",
                        "train_test_accord_chaim/binary_allergy_ts/sesame/train_val_set_ALL_ALLERGY_microbiome.csv",
                        "train_test_accord_chaim/binary_allergy_ts/sesame/test_set_ALL_ALLERGY_microbiome.csv"
                    )
        if tag_name == 'milk':
            if after_split == False:
                tag = pd.read_csv("Data/Allergy/Allergy_all_times_all_kinds/milk_vs_all.csv.csv", index_col=0)
            if after_split == True:
                tag_train = pd.read_csv("train_test_accord_chaim/binary_allergy_ts/milk/train_val_set_ALL_ALLERGY_tags.csv",
                                        index_col=0)
                tag_test = pd.read_csv("train_test_accord_chaim/binary_allergy_ts/milk/test_set_ALL_ALLERGY_tags.csv", index_col=0)

        if tag_name == 'nuts':
            if after_split == False:
                tag = pd.read_csv("Data/Allergy/Allergy_all_times_all_kinds/nuts_vs_all.csv.csv", index_col=0)
            if after_split == True:
                tag_train = pd.read_csv("train_test_accord_chaim/binary_allergy_ts/nuts/train_val_set_ALL_ALLERGY_tags.csv",
                                        index_col=0)
                tag_test = pd.read_csv("train_test_accord_chaim/binary_allergy_ts/nuts/test_set_ALL_ALLERGY_tags.csv", index_col=0)

        if tag_name == 'peanuts':
            if after_split == False:
                tag = pd.read_csv("Data/Allergy/Allergy_all_times_all_kinds/peanuts_vs_all.csv", index_col=0)
            if after_split == True:
                tag_train = pd.read_csv("train_test_accord_chaim/binary_allergy_ts/peanuts/train_val_set_ALL_ALLERGY_tags.csv",
                                        index_col=0)
                tag_test = pd.read_csv("train_test_accord_chaim/binary_allergy_ts/peanuts/test_set_ALL_ALLERGY_tags.csv",
                                       index_col=0)

        if tag_name == 'sesame':
            if after_split == False:
                tag = pd.read_csv("Data/Allergy/Allergy_all_times_all_kinds/sesame_vs_all.csv", index_col=0)
            if after_split == True:
                tag_train = pd.read_csv("train_test_accord_chaim/binary_allergy_ts/sesame/train_val_set_ALL_ALLERGY_tags.csv",
                                        index_col=0)
                tag_test = pd.read_csv("train_test_accord_chaim/binary_allergy_ts/sesame/test_set_ALL_ALLERGY_tags.csv", index_col=0)

    if name_of_dataset == "PNAS":
        _map = pd.read_csv("Data_TS/PNAS/sample_times.csv", index_col=0)
        group = _map["Group"]
        del _map["GestationalDayOfDelivery"]
        if D_mode != "1D":
            del _map["Group"]
        if after_split == False:
            tag = pd.read_csv("Data_TS/PNAS/tag.csv", index_col=0)

        else:
            tag_train = pd.read_csv("train_test_accord_chaim/PNAS/train_val_set_PNAS_tags.csv", index_col=0)
            tag_test = pd.read_csv("train_test_accord_chaim/PNAS/test_set_PNAS_tags.csv", index_col=0)

        task = "class"
        if D_mode == "1D":

            if after_split == False:
                otu = pd.read_csv("Data_TS/PNAS/otus_tax_7_sub_pca_log.csv", index_col=0)
            else:

                otu_train = pd.read_csv(
                    "train_test_accord_chaim/PNAS/train_val_set_PNAS_microbiome.csv", index_col=0)
                otu_test = pd.read_csv("train_test_accord_chaim/PNAS/test_set_PNAS_microbiome.csv",
                                       index_col=0)
                group = _map["Group"].loc[otu_train.index]
                del _map["Group"]
        else:
            path_of_2D_matrix = "2D_otus_dendogram_ordered/PNAS"
            if after_split == False:
                otu = pd.read_csv("2D_otus_dendogram_ordered/PNAS/0_fixed_ordered_n_pnas_otu_sub_pca_log_tax_7.csv",
                                  index_col=0)
            else:
                otu_train, otu_test = after_split_from_pd(
                    "2D_otus_dendogram_ordered/PNAS/0_fixed_ordered_n_pnas_otu_sub_pca_log_tax_7.csv",
                    "train_test_accord_chaim/PNAS/train_val_set_PNAS_microbiome.csv",
                    "train_test_accord_chaim/PNAS/test_set_PNAS_microbiome.csv"
                )

    if name_of_dataset == "multi_allergy":
        _map = pd.read_csv("Data/Allergy/Allergy_all_times_all_kinds/mapping.csv", index_col=0)
        group = _map["Group"]

        if D_mode != "1D":
            del _map["Group"]
        if after_split == False:
            tag = pd.read_csv("Data/Allergy/Allergy_all_times_all_kinds/tag.csv", index_col=0)

        else:
            tag_train = pd.read_csv("train_test_accord_chaim/multi_allergy/train_val_set_ALL_ALLERGY_tags.csv", index_col=0)
            tag_test = pd.read_csv("train_test_accord_chaim/multi_allergy/test_set_ALL_ALLERGY_tags.csv", index_col=0)

        task = "class"
        if D_mode == "1D":

            if after_split == False:
                otu = pd.read_csv("Data/Allergy/Allergy_all_times_all_kinds/otus.csv", index_col=0)
            else:

                otu_train = pd.read_csv(
                    "train_test_accord_chaim/multi_allergy/train_val_set_ALL_ALLERGY_microbiome.csv", index_col=0)
                otu_test = pd.read_csv("train_test_accord_chaim/multi_allergy/test_set_ALL_ALLERGY_microbiome.csv",
                                       index_col=0)
                group = _map["Group"].loc[otu_train.index]
                del _map["Group"]
        else:
            path_of_2D_matrix = "2D_otus_dendogram_ordered/Allergy_all_kind_all_times"
            if after_split == False:
                otu = pd.read_csv("2D_otus_dendogram_ordered/Allergy_all_kind_all_times/0_fixed_ordered_n_all_otu_sub_pca_log_tax_7.csv",
                                  index_col=0)
            else:
                otu_train, otu_test = after_split_from_pd(
                    "2D_otus_dendogram_ordered/Allergy_all_kind_all_times/0_fixed_ordered_n_all_otu_sub_pca_log_tax_7.csv",
                    "train_test_accord_chaim/multi_allergy/train_val_set_ALL_ALLERGY_microbiome.csv",
                    "train_test_accord_chaim/multi_allergy/test_set_ALL_ALLERGY_microbiome.csv"
                )

    if name_of_dataset == "T2D":
        tag = pd.read_csv("Data/PopPhy_data/T2D/tag.csv", index_col=0)

        group = tag.index
        # del map["Group"]
        # del map["Tag"]
        biomarkers = _map
        task = "class"

        if D_mode == "IEEE":
            otu = pd.read_csv(
                "Data/PopPhy_data/T2D/T2D_otu_mean_relative_rare_bact_5_tax_7_only_after_git_preprocess.csv",
                index_col=0)
            path_of_2D_matrix = "2D_OTU_IEEE/PopPhy_data/T2D"

        elif D_mode == "1D":
            otu = pd.read_csv(
                "Data/PopPhy_data/T2D/T2D_otu_mean_log_rare_bact_5_tax_7_only_after_git_preprocess.csv",
                index_col=0)

        else:
            otu = pd.read_csv(
                "2D_otus_dendogram_ordered/PopPhy_data/T2D/0_fixed_ordered_T2D_otu_sub_pca_log_tax_7.csv",
                index_col=0)
            path_of_2D_matrix = "2D_otus_dendogram_ordered/PopPhy_data/T2D"

    if name_of_dataset == "Gastro_vs_oral":
        tag = pd.read_csv("Data/Gastro_vs_oral/Data_for_learning/tag_oral_not_oral.csv", index_col=0)
        _map = pd.read_csv("Data/Gastro_vs_oral/Data_for_learning/map.csv", index_col=0)
        group = _map["Group"]
        del _map["Group"]
        del _map["Tag"]
        biomarkers = _map
        task = "class"

        if D_mode == "IEEE":
            otu = pd.read_csv(
                "Data/Gastro_vs_oral/Gastro_vs_oral_otu_sum_relative_rare_bact_5_tax_7_only_after_git_preprocess.csv",
                index_col=0)
            path_of_2D_matrix = "2D_OTU_IEEE/Gastro_vs_oral"
        elif D_mode == "1D":
            otu = pd.read_csv("Data/Gastro_vs_oral/Data_for_learning/otu_tax_7_log_sub_pca.csv", index_col=0)

        else:
            otu = pd.read_csv(
                "2D_otus_dendogram_ordered/Gastro_vs_oral/0_fixed_ordered_Global_vs_oral_otu_sub_pca_log_tax_7.scv",
                index_col=0)
            path_of_2D_matrix = "2D_otus_dendogram_ordered/Gastro_vs_oral"

    if otu is None:
        otu = otu_train
    if with_map == False:
        input_dim = len(otu.columns)
    else:
        input_dim = len(otu.columns) + len(biomarkers.columns)

    os.chdir(org_path)

    try:
        if type(tag) is pd.DataFrame:
            tag = tag[tag.columns[0]]
    except:
        if type(tag_train) is pd.DataFrame:
            tag_train = tag_train[tag_train.columns[0]]
        if type(tag_test) is pd.DataFrame:
            tag_test = tag_test[tag_test.columns[0]]
    if D_mode != "1D":
        if name_of_dataset == "GDM_T2_T3":
            input_dim = (7, input_dim)
        else:
            input_dim = (8, input_dim)

    if biomarkers is not None and biomarkers.shape[1] == 0:
        biomarkers = None

    if group is None:
        group = otu.index
    if after_split == False:
        return otu, tag, group, biomarkers, path_of_2D_matrix, input_dim, task
    else:
        return otu_train, otu_test, tag_train, tag_test, group, biomarkers, path_of_2D_matrix, input_dim, task


if __name__ == '__main__':
    ############################
    # BGU T0 T18
    ###########################
    # otu,tag,map,input_dim,task = load_nni_data("BGU", dendogram=False,tag_name="dsc",with_map=True)
    # otu, tag, map, input_dim, task = load_nni_data("BGU", dendogram=True, tag_name="dsc", with_map=True)
    # otu, tag, map, input_dim, task = load_nni_data("BGU", dendogram=True, tag_name="dsc", with_map=False)
    # otu, tag, map, input_dim, task = load_nni_data("BGU", dendogram=False, tag_name="ssc", with_map=True)
    # otu, tag, map, input_dim, task = load_nni_data("BGU", dendogram=True, tag_name="vat", with_map=False)
    # otu, tag, map, input_dim, task = load_nni_data("BGU", dendogram=True, tag_name="li", with_map=False)
    ##########################################################################################################
    ############################
    # IBD
    #########################
    # otu, tag, map, input_dim, task = load_nni_data("IBD", dendogram=True, tag_name="CD", with_map=False)
    # otu, tag, map, input_dim, task = load_nni_data("IBD", dendogram=False, tag_name="IBD", with_map=False)
    #############################################################################################################
    #####################
    # GDM 2 3
    #########################
    # otu, tag, map, input_dim, task = load_nni_data("GDM_T2_T3", dendogram=False, tag_name=None, with_map=False)
    #######################################################################################################
    ##################
    # GASTRO VS ORAL
    ##################
    otu, tag, _map, input_dim, task = load_nni_data("Gastro_vs_oral", dendogram=True, tag_name=None, with_map=False)
    ################################################################################################################
    # ALLERGY
    ##############################
    otu, tag, _map, input_dim, task = load_nni_data("Allergy_MILK_T0", dendogram=False, tag_name=None, with_map=False)

    x = 5

    # if name_of_dataset == "BGU_T0":
    #     biomarkers = pd.read_csv("Data/OSHRIT_all_biomarkers_features_scaled.csv", index_col=0)
    #
    #     if tag_name == "dsc":
    #         tag = pd.read_csv("Data/BGU_T0/DSC/tag.csv", index_col=0)
    #         if D_mode == "IEEE":
    #             otu = pd.read_csv("Data/BGU_otu_sum_relative_rare_bact_5_tax_7.csv", index_col=0)
    #             path_of_2D_matrix = "2D_otus_IEEE_BGU"
    #         elif D_mode == "1D":
    #             otu = pd.read_csv("Data/BGU_T0/DSC/otus_subpca_tax7_log.csv", index_col=0)
    #
    #
    #         else:
    #             otu = pd.read_csv("2D_otus_dendogram_ordered/BGU_T0/DSC/0_fixed_ordered_BGU_otu_sub_pca_log_tax_7.csv",
    #                               index_col=0)
    #             path_of_2D_matrix = "2D_otus_dendogram_ordered/BGU_T0/DSC"
    #
    #     elif tag_name == "ssc":
    #         tag = pd.read_csv("Data/BGU_T0/SSC/tag.csv", index_col=0)
    #
    #         if D_mode == "IEEE":
    #             otu = pd.read_csv("Data/BGU_otu_sum_relative_rare_bact_5_tax_7.csv", index_col=0)
    #             path_of_2D_matrix = "2D_otus_IEEE_BGU"
    #         elif D_mode == "1D":
    #             otu = pd.read_csv("Data/BGU_T0/SSC/otus_subpca_tax7_log.csv", index_col=0)
    #
    #         else:
    #             otu = pd.read_csv("2D_otus_dendogram_ordered/BGU_T0/SSC/0_fixed_ordered_BGU_otu_sub_pca_log_tax_7.csv",
    #                               index_col=0)
    #             path_of_2D_matrix = "2D_otus_dendogram_ordered/BGU_T0/SSC"
    #
    #     elif tag_name == "vat":
    #         tag = pd.read_csv("Data/BGU_T0/VAT/tag.csv", index_col=0)
    #         if D_mode == "IEEE":
    #             otu = "Data/BGU_otu_sum_relative_rare_bact_5_tax_7.csv"
    #             path_of_2D_matrix = "2D_otus_IEEE_BGU"
    #         elif D_mode == "1D":
    #             otu = pd.read_csv("Data/BGU_T0/SSC/otus_subpca_tax7_log.csv", index_col=0)
    #
    #         else:
    #             otu = pd.read_csv("2D_otus_dendogram_ordered/BGU_T0/VAT/0_fixed_ordered_BGU_otu_sub_pca_log_tax_7.csv",
    #                               index_col=0)
    #             path_of_2D_matrix = "2D_otus_dendogram_ordered/BGU_T0/VAT"
    #
    #     elif tag_name == 'li':
    #         tag = pd.read_csv("Data/BGU_T0/LI/tag.csv", index_col=0)
    #         if D_mode == "IEEE":
    #             otu = "Data/BGU_otu_sum_relative_rare_bact_5_tax_7.csv"
    #             path_of_2D_matrix = "2D_otus_IEEE_BGU"
    #         elif D_mode == "1D":
    #             otu = pd.read_csv("Data/BGU_T0/SSC/otus_subpca_tax7_log.csv", index_col=0)
    #
    #         else:
    #             otu = pd.read_csv("2D_otus_dendogram_ordered/BGU_T0/LI/0_fixed_ordered_BGU_otu_sub_pca_log_tax_7.csv",
    #                               index_col=0)
    #             path_of_2D_matrix = "2D_otus_dendogram_ordered/BGU_T0/LI"
    #
    #     else:
    #         raise AttributeError("No tag file specified. Choose from: dsc, ssc, vat, li.")
    #
    #     task = "reg"
    # if name_of_dataset == "BGU":
    #     if tag_name == "dsc":
    #         tag = pd.read_csv("Data/BGU_T0_T18/DSC/tag.csv", index_col=0)
    #         _map = pd.read_csv("Data/BGU_T0_T18/DSC/map.csv", index_col=0)
    #         group = _map["Group"]
    #         # group = tag.index
    #         del _map["Group"]
    #         biomarkers = _map
    #         if D_mode == "IEEE":
    #             otu = pd.read_csv("Data/BGU_otu_sum_relative_rare_bact_5_tax_7.csv", index_col=0)
    #             path_of_2D_matrix = "2D_otus_IEEE_BGU"
    #         elif D_mode == "1D":
    #             otu = pd.read_csv("Data/BGU_T0/SSC/otus_subpca_tax7_log.csv", index_col=0)
    #
    #         else:
    #             otu = pd.read_csv("2D_otus_dendogram_ordered/BGU/OTUS/0_fixed_ordered_GBGU_otu_sub_pca_log_tax_7.scv",
    #                               index_col=0)
    #             path_of_2D_matrix = "2D_otus_dendogram_ordered/BGU/OTUS"
    #
    #     elif tag_name == "ssc":
    #         _map = pd.read_csv("Data/BGU_T0_T18/SSC/map.csv", index_col=0)
    #         group = _map["Group"]
    #         del _map["Group"]
    #         biomarkers = _map
    #         tag = pd.read_csv("Data/BGU_T0_T18/SSC/tag.csv", index_col=0)
    #         if D_mode == "IEEE":
    #             otu = pd.read_csv("Data/BGU_otu_sum_relative_rare_bact_5_tax_7.csv", index_col=0)
    #             path_of_2D_matrix = "2D_otus_IEEE_BGU"
    #         elif D_mode == "1D":
    #             otu = pd.read_csv("Data/BGU_T0/SSC/otus_subpca_tax7_log.csv", index_col=0)
    #
    #         else:
    #             otu = pd.read_csv("2D_otus_dendogram_ordered/BGU/OTUS/0_fixed_ordered_GBGU_otu_sub_pca_log_tax_7.scv",
    #                               index_col=0)
    #             path_of_2D_matrix = "2D_otus_dendogram_ordered/BGU/OTUS"
    #
    #     elif tag_name == "vat":
    #         _map = pd.read_csv("Data/BGU_T0_T18/VAT/map.csv", index_col=0)
    #         group = map["Group"]
    #         del _map["Group"]
    #         biomarkers = _map
    #         tag = pd.read_csv("Data/BGU_T0_T18/VAT/tag.csv", index_col=0)
    #         if D_mode == "IEEE":
    #             otu = pd.read_csv("Data/BGU_otu_sum_relative_rare_bact_5_tax_7.csv", index_col=0)
    #             path_of_2D_matrix = "2D_otus_IEEE_BGU"
    #         elif D_mode == "1D":
    #             otu = pd.read_csv("Data/BGU_T0/SSC/otus_subpca_tax7_log.csv", index_col=0)
    #
    #         else:
    #             otu = pd.read_csv("2D_otus_dendogram_ordered/BGU/OTUS/0_fixed_ordered_GBGU_otu_sub_pca_log_tax_7.scv",
    #                               index_col=0)
    #             path_of_2D_matrix = "2D_otus_dendogram_ordered/BGU/OTUS"
    #
    #
    #     elif tag_name == 'li':
    #         _map = pd.read_csv("Data/BGU_T0_T18/LI/map.csv", index_col=0)
    #         group = _map["Group"]
    #         del _map["Group"]
    #         biomarkers = _map
    #         tag = pd.read_csv("Data/BGU_T0_T18/LI/tag.csv", index_col=0)
    #         if D_mode == "IEEE":
    #             otu = pd.read_csv("Data/BGU_otu_sum_relative_rare_bact_5_tax_7.csv", index_col=0)
    #             path_of_2D_matrix = "2D_otus_IEEE_BGU"
    #         elif D_mode == "1D":
    #             otu = pd.read_csv("Data/BGU_T0/SSC/otus_subpca_tax7_log.csv", index_col=0)
    #
    #         else:
    #             otu = pd.read_csv("2D_otus_dendogram_ordered/BGU/OTUS/0_fixed_ordered_GBGU_otu_sub_pca_log_tax_7.scv",
    #                               index_col=0)
    #             path_of_2D_matrix = "2D_otus_dendogram_ordered/BGU/OTUS"
    #
    #     else:
    #         raise AttributeError("No tag file specified. Choose from: dsc, ssc, vat, li.")
    #     # normalize tag:
    #     tag = (tag - tag.mean()) / tag.std()
    #     task = "reg"
