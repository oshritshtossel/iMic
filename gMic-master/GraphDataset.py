import networkx as nx
from torch.utils.data import Dataset
from torch import Tensor
from MicrobiomeDataset import MicrobiomeDataset
from create_microbiome_graphs import CreateMicrobiomeGraphs


class GraphDataset(Dataset):
    def __init__(self, data_file_path, tag_file_path, mission):
        super(GraphDataset, self).__init__()
        # for dataset handling
        self.microbiome_dataset = MicrobiomeDataset(data_file_path, tag_file_path)
        microbiome_df = self.microbiome_dataset.microbiome_df
        # for graphs' creation
        self.create_microbiome_graphs = CreateMicrobiomeGraphs(microbiome_df)
        self.samples_len = microbiome_df.shape[0]
        self.dataset_dict = {}
        self.mission = mission

    def set_dataset_dict(self, **kwargs):
        dataset_dict = {}
        for i in range(self.samples_len):
            graph = self.create_microbiome_graphs.get_graph(i)
            dataset_dict[i] = {'graph': graph,
                               'label': self.microbiome_dataset.get_label(i),
                               'values_on_leaves': self.microbiome_dataset.get_leaves_values(i),
                               'values_on_nodes': self.create_microbiome_graphs.get_values_on_nodes_ordered_by_nodes(graph),
                               'adjacency_matrix': nx.adjacency_matrix(graph).todense()}
        return dataset_dict

    def get_joint_nodes(self):
        return self.create_microbiome_graphs.find_common_nodes().keys()

    def update_graphs(self, **kwargs):
        # self.create_microbiome_graphs.create_graphs_with_common_nodes(union_train_and_test)
        self.dataset_dict = self.set_dataset_dict(**kwargs)  # create dataset dict only after we added the missing nodes to the graph

    def get_all_groups(self):
        return self.microbiome_dataset.groups

    def get_leaves_number(self):
        return self.microbiome_dataset.get_leaves_number()

    def get_vector_size(self):
        return self.create_microbiome_graphs.get_vector_size()

    def nodes_number(self):
        return self.create_microbiome_graphs.nodes_number()

    def __len__(self):
        return self.samples_len

    def __getitem__(self, index):
        index_value = self.dataset_dict[index]
        label = index_value['label']
        if self.mission == "just_values":
            values = index_value['values_on_leaves']
            adjacency_matrix = index_value['adjacency_matrix']
        elif self.mission == "just_graph" or self.mission == "graph_and_values":
            values = index_value['values_on_nodes']
            adjacency_matrix = index_value['adjacency_matrix']
        return Tensor(values), Tensor(adjacency_matrix), label
