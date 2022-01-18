import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import argparse
import json
import logging
import csv
import nni
import numpy as np
import torch
from GraphDataset import GraphDataset
from train_test_val_ktimes import TrainTestValKTimes
from MyTasks import *
from MyDatasets import *
# import warnings
# warnings.simplefilter(action='ignore', category=UserWarning)

LOG = logging.getLogger('nni_logger')
K = 10  # For k-cross-validation
datasets_dict = {"bw": MyDatasets.bw_files, "IBD_Chrone": MyDatasets.ibd_chrone_files,
                 "male_female": MyDatasets.male_vs_female,
                 "nut": MyDatasets.nut, "peanut": MyDatasets.peanut, "nugent": MyDatasets.nugent,
                 "milk": MyDatasets.allergy_milk_no_controls, "Cirrhosis": MyDatasets.cirrhosis_files,
                "IBD": MyDatasets.ibd_files}

tasks_dict = {1: MyTasks.just_values, 2: MyTasks.just_graph_structure, 3: MyTasks.values_and_graph_structure}


class Main:
    def __init__(self, dataset_name, task_number, RECEIVED_PARAMS, device, nni_mode=False, plot_figures=False):
        self.dataset_name = dataset_name
        self.task_number = task_number
        self.RECEIVED_PARAMS = RECEIVED_PARAMS
        self.device = device
        self.nni_mode = nni_mode
        self.plot_figures = plot_figures

    def create_dataset(self, data_file_path, tag_file_path, mission):
        cur_dataset = GraphDataset(data_file_path, tag_file_path, mission)
        return cur_dataset

    def turn_on_train(self):
        my_tasks = MyTasks(tasks_dict, self.dataset_name)
        my_datasets = MyDatasets(datasets_dict)

        directory_name, mission, params_file_path = my_tasks.get_task_files(self.task_number)
        result_directory_name = os.path.join(directory_name, "Result_After_Proposal")
        train_data_file_path, train_tag_file_path, test_data_file_path, test_tag_file_path = \
            my_datasets.microbiome_files(self.dataset_name)

        print("Training-Validation Sets Graphs")
        train_val_dataset = self.create_dataset(train_data_file_path, train_tag_file_path, mission)
        print("Test set Graphs")
        test_dataset = self.create_dataset(test_data_file_path, test_tag_file_path, mission)

        train_val_dataset.update_graphs()
        test_dataset.update_graphs()

        trainer_and_tester = TrainTestValKTimes(self.RECEIVED_PARAMS, self.device, train_val_dataset, test_dataset,
                                                result_directory_name, nni_flag=self.nni_mode)
        train_metric, val_metric, test_metric, min_train_val_metric = trainer_and_tester.train_group_k_cross_validation(k=K)
        return train_metric, val_metric, test_metric, min_train_val_metric


def set_arguments():
    parser = argparse.ArgumentParser(description='Main script of all models')
    parser.add_argument("--dataset", help="Dataset name", default="IBD", type=str)
    parser.add_argument("--task_number", help="Task number", default=1, type=int)
    parser.add_argument("--device_num", help="Cuda Device Number", default=1, type=int)
    parser.add_argument("--nni", help="is nni mode", default=0, type=int)
    return parser


def results_dealing(train_metric, val_metric, test_metric, min_train_val_metric, nni_flag, RECEIVED_PARAMS, result_file_name):
    mean_train_metric = np.average(train_metric)
    std_train_metric = np.std(train_metric)
    mean_min_train_val_metric = np.average(min_train_val_metric)
    std_min_train_val_metric = np.std(min_train_val_metric)
    mean_val_metric = np.average(val_metric)
    std_val_metric = np.std(val_metric)
    mean_test_metric = np.average(test_metric)
    std_test_metric = np.std(test_metric)

    if nni_flag:
        LOG.debug("\n \nMean Validation Set AUC: ", mean_min_train_val_metric, " +- ", std_min_train_val_metric)
        LOG.debug("\nMean Test Set AUC: ", mean_test_metric, " +- ", std_test_metric)
        nni.report_intermediate_result(mean_test_metric)
        nni.report_final_result(mean_min_train_val_metric)
    else:
        result_file_name = f"{result_file_name}_val_mean_{mean_val_metric:.3f}_test_mean_{mean_test_metric:.3f}.csv"
        f = open(result_file_name, 'w', newline='')
        writer = csv.writer(f)
        writer.writerow([","] + [f"Run{i}" for i in range(len(val_metric))] + ["", "Mean+-std"])
        writer.writerow(['val_auc'] + val_metric + ["", str(mean_val_metric) + "+-" + str(std_val_metric)])
        writer.writerow(['test_auc'] + test_metric + ["", str(mean_test_metric) + "+-" + str(std_test_metric)])
        writer.writerow(['train_auc'] + train_metric + ["", str(mean_train_metric) + "+-" + str(std_train_metric)])
        writer.writerow([])
        writer.writerow([])
        for key, value in RECEIVED_PARAMS.items():
            writer.writerow([key, value])
        f.close()

    print("\n \nMean minimum Validation and Train Sets AUC: ", mean_min_train_val_metric, " +- ", std_min_train_val_metric)
    print("Mean Validation Set AUC: ", mean_val_metric, " +- ", std_val_metric)
    print("Mean Test Set AUC: ", mean_test_metric, " +- ", std_test_metric)
    print("Mean Train Set AUC: ", mean_train_metric, " +- ", std_train_metric)


def run_all_dataset(mission_number, cuda_number, nni_flag):
    for dataset_name in datasets_dict.keys():
        try:
            run_regular(dataset_name, mission_number, cuda_number, nni_flag)
        except Exception as e:
            print(e)


def run_all_datasets_missions(cuda_number, nni_flag):
    for mission_number in tasks_dict.keys():
        run_all_dataset(mission_number, cuda_number, nni_flag)


def run_regular(dataset_name, mission_number, cuda_number, nni_flag):
    my_tasks = MyTasks(tasks_dict, dataset_name)
    directory_name, mission, params_file_path = my_tasks.get_task_files(mission_number)
    if nni_flag:
        RECEIVED_PARAMS = nni.get_next_parameter()
    else:
        RECEIVED_PARAMS = json.load(open(params_file_path, 'r'))
    RECEIVED_PARAMS["learning_rate"] = np.float64(RECEIVED_PARAMS["learning_rate"])
    RECEIVED_PARAMS["dropout"] = np.float64(RECEIVED_PARAMS["dropout"])
    RECEIVED_PARAMS["regularization"] = np.float64(RECEIVED_PARAMS["regularization"])
    RECEIVED_PARAMS["train_frac"] = np.float64(RECEIVED_PARAMS["train_frac"])
    RECEIVED_PARAMS["test_frac"] = np.float64(RECEIVED_PARAMS["test_frac"])

    print("Dataset", dataset_name)
    device = f"cuda:{cuda_number}" if torch.cuda.is_available() else "cpu"
    print("Device", device)
    print("Task", mission)
    main_runner = Main(dataset_name, mission_number, RECEIVED_PARAMS, device, nni_mode=nni_flag)
    train_metric, val_metric, test_metric, min_train_val_metric = main_runner.turn_on_train()
    result_file_name = f"{dataset_name}_{mission}"
    results_dealing(train_metric, val_metric, test_metric, min_train_val_metric, nni_flag, RECEIVED_PARAMS, result_file_name)


if __name__ == '__main__':
    try:
        parser = set_arguments()
        args = parser.parse_args()
        dataset_name = args.dataset

        mission_number = args.task_number
        cuda_number = args.device_num
        nni_flag = False if args.nni == 0 else True

        for dataset_name in list(datasets_dict.keys()):
            for mission_number in [1,2,3]:
                run_regular(dataset_name, mission_number, cuda_number, nni_flag)

    except Exception as e:
        LOG.exception(e)
        raise
