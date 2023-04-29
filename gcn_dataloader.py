import torch
import numpy
import json
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from transformers import AutoTokenizer
import numpy as np

class SceneGraphDataset(Dataset):
    def __init__(self, data_dir, graph_dir, image_dir, vecs_dir, transform=None, pre_transform=None, pre_filter=None):
        self.graph_list = []
        with open(graph_dir) as infile:
            for line in infile:
                graphs = json.loads(line)
                # self.graph_list.append(graphs)
        # print(len(graphs))
        # print(len(graphs))
        # import ipdb; ipdb.set_trace()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
        # filtered_graph_dict = {}
        # for example in self.graph_list:
        #     # try:
        #     #     graph0 = graph[graphs[index]['identifier'][:-2]+"-img0.png"]
        #     #     graph1 = graph[graphs[index]['identifier'][:-2]+"-img1.png"]
        #     #     filtered_graph_dict[graphs[index]['identifier'][:-2]+"-img0.png"] = graph0
        #     #     filtered_graph_dict[graphs[index]['identifier'][:-2]+"-img1.png"] = graph1
        #     # except:
        #     #     continue  
        #     # print(example)
        #     import ipdb; ipdb.set_trace()
        # print(len(self.graph_list))
        self.graph_data = graphs
        # print(self.graph_data)
        self.image_dir = image_dir

        self.data = []
        with open(data_dir) as infile:
            for line in infile:
                example = json.loads(line)
                self.data.append(example)

        self.filtered_examples = []
        self.filtered_vecs = []

        for i, ann in enumerate(self.data):
            image_1_path = ann['identifier'][:-2]+"-img0.png"
            image_2_path = ann['identifier'][:-2]+"-img1.png"
            #load from npy file
            vec = np.load(vecs_dir+"/"+str(i)+".npy")
            #check if graph exists for both images
            if image_1_path in self.graph_data and image_2_path in self.graph_data and len(self.graph_data[image_1_path])>0 and len(self.graph_data[image_2_path])>0:
                self.filtered_examples.append(ann)
                self.filtered_vecs.append(vec)

#         print(len(self.filtered_examples), len(self.data))
        # import ipdb; ipdb.set_trace()

        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large")
        # print(len(self.filtered_examples))
    
    def __len__(self):
        return len(self.filtered_examples)

    def filter_graph(self, graph):
        graph_fil = []
        for edge in graph:
            if edge['score'] >= 1e-4:
                graph_fil.append(edge)
        return graph_fil

    def get_unique_nodes(self, graph):
        unique_nodes = {}
        node_index = 0
        for edge in graph:
            x_1 = edge['subject']
            y_1 = edge['object']
            e = edge['relation']
            if x_1 not in unique_nodes:
                unique_nodes[x_1] = node_index
                node_index += 1
            if y_1 not in unique_nodes:
                unique_nodes[y_1] = node_index
                node_index += 1
        return unique_nodes

    def get_x_list(self, unique_nodes_1, unique_nodes_2):
        x_list = []
        for node in unique_nodes_1:
            # x_list.append(node)
            token_x = self.tokenizer(node, return_tensors="pt", padding="max_length", max_length=64, truncation=True)
            input_ids_x = token_x["input_ids"]
            x_list.append(input_ids_x.squeeze(0))

        for node in unique_nodes_2:
            # x_list.append(node)
            token_x = self.tokenizer(node, return_tensors="pt", padding="max_length", max_length=64, truncation=True)
            input_ids_x = token_x["input_ids"]
            x_list.append(input_ids_x.squeeze(0))
        return x_list


    def get_self_edges(self, graph, unique_nodes):
        edge_list = []
        edge_attr = []
        for edge in graph:
            x_1 = edge['subject']
            y_1 = edge['object']
            e = edge['relation']
            x_id = unique_nodes[x_1]
            y_id = unique_nodes[y_1]
            edge_list.append([x_id, y_id])
            token_e = self.tokenizer(e, return_tensors="pt", padding="max_length", max_length=64, truncation=True)
            input_ids_e = token_e["input_ids"]
            edge_attr.append(input_ids_e.squeeze(0))
        return edge_list, edge_attr

    def get_cross_edges(self, unique_nodes_1, unique_nodes_2):
        edge_list = []
        edge_attr = []
        for n1 in unique_nodes_1:
            for n2 in unique_nodes_2:
                edge_list.append([unique_nodes_1[n1], unique_nodes_2[n2]])

                txt_cross = "cross_edge"
                token_e = self.tokenizer(txt_cross, return_tensors="pt", padding="max_length", max_length=64, truncation=True)
                input_ids_e = token_e["input_ids"]
                edge_attr.append(input_ids_e.squeeze(0))

                #add other direction edge as well
                edge_list.append([unique_nodes_2[n2], unique_nodes_1[n1]])
                edge_attr.append(input_ids_e.squeeze(0))
        return edge_list, edge_attr
       

    def __getitem__(self, index):
        dict_values = {}
        ann = self.filtered_examples[index]
        text = ann['sentence']
        label = ann['label']
        identifier = ann['identifier']
        dict_values["identifier"] = identifier
        # print(label)
        if label == "True":
            dict_values["label"] = 1
        else:
            dict_values["label"] = 0
        inputs_dict = self.tokenizer(text, return_tensors="pt", padding="max_length", max_length=64, truncation=True)
        input_ids = inputs_dict["input_ids"]
        attention_mask = inputs_dict["attention_mask"]
        dict_values["input_ids"] = input_ids.squeeze(0)
        dict_values["attention_mask"] = attention_mask.squeeze(0)
        dict_values["u"] = torch.Tensor(self.filtered_vecs[index])
        #get graphs
        image_1_path = ann['identifier'][:-2]+"-img0.png"
        image_2_path = ann['identifier'][:-2]+"-img1.png"
        graph_1 = self.graph_data[image_1_path]
        graph_2 = self.graph_data[image_2_path]
        # import ipdb; ipdb.set_trace()

        # print(len(graph_1), len(graph_2))

        # graph_1 = self.filter_graph(graph_1)
        # graph_2 = self.filter_graph(graph_2)   
        # print(len(graph_1), len(graph_2))
        # import ipdb; ipdb.set_trace()
        x_list = []
        edge_list = []
        edge_attr = []
        unique_nodes_1 = self.get_unique_nodes(graph_1)
        unique_nodes_2 = self.get_unique_nodes(graph_2)
        for node in unique_nodes_2:
            unique_nodes_2[node] += len(unique_nodes_1)


        x_list = self.get_x_list(unique_nodes_1, unique_nodes_2)
        x = torch.stack(x_list)

        edge_list_1, edge_attr_1 = self.get_self_edges(graph_1, unique_nodes_1)
        edge_list_2, edge_attr_2 = self.get_self_edges(graph_2, unique_nodes_2)
        edge_list.extend(edge_list_1)
        edge_list.extend(edge_list_2)
        edge_attr.extend(edge_attr_1)
        edge_attr.extend(edge_attr_2)

        edge_list_cross, edge_attr_cross = self.get_cross_edges(unique_nodes_1, unique_nodes_2)
        edge_list.extend(edge_list_cross)
        edge_attr.extend(edge_attr_cross)


        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attr)
        # print(edge_index)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        dict_values["graph"] = data
        return dict_values

        


        
        
        
    
