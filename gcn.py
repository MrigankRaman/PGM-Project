from torch_scatter import scatter_mean, scatter_sum, scatter_max
from torch_scatter.composite import scatter_softmax
from gcn_dataloader import SceneGraphDataset
from torch_geometric.loader import DataLoader
from layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

class EdgeModel(torch.nn.Module):
    def __init__(self, edge_in_dim, hidden_dim, edge_out_dim, edge_weight_dropout, u_dim, ablation):
        super(EdgeModel, self).__init__()
        self.ablation = ablation
        if "no_vec" in ablation:
            self.edge_mlp = MLP(3*edge_in_dim, hidden_dim, 768//2,2, 0.2, batch_norm=False, layer_norm=True)
            self.weight_mlp = MLP(edge_in_dim, hidden_dim, 1,  # todo: avoid hard-coded numbers
                              1, edge_weight_dropout, batch_norm=False, layer_norm=True)
        else:
            self.edge_mlp = MLP(3*edge_in_dim + u_dim, hidden_dim, 768//2,
                            2, 0.2, batch_norm=False, layer_norm=True)
            self.weight_mlp = MLP(edge_in_dim + u_dim, hidden_dim, 1,  # todo: avoid hard-coded numbers
                              1, edge_weight_dropout, batch_norm=False, layer_norm=True)
        # self.wt_transform = nn.Linear(edge_in_dim - 128, 128)
        self.dropout = nn.Dropout(0.1)

    def forward(self, src, dest, edge_attr, u, edge_batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        # import ipdb
        # ipdb.set_trace()
        if "no_vec" in self.ablation:
            out = torch.cat([src, dest, edge_attr], 1)
            out_1 = edge_attr
        else:
            out = torch.cat([src, dest, edge_attr, u[edge_batch]], 1)
            out_1 = torch.cat([edge_attr, u[edge_batch]], 1)
        wts = self.weight_mlp(out_1)  # wts: [#edges, 1]
        unnormalized_wts = wts
        wts = scatter_softmax(wts.squeeze(1), edge_batch, dim=0)
        normalized_wts = wts.unsqueeze(1)
        return self.edge_mlp(out), unnormalized_wts, normalized_wts


class NodeModel(torch.nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim_1, hidden_dim_2, u_dim, ablation):
        super(NodeModel, self).__init__()
        self.ablation = ablation
        mlp_1_in_dim = node_in_dim + edge_in_dim
        if "no_vec" in self.ablation:
            mlp_2_in_dim = 256 + node_in_dim
        else:
            mlp_2_in_dim = 256 + node_in_dim + u_dim
        self.message_mlp = MLP(mlp_1_in_dim, hidden_dim_1, 256,
                               2, 0.2, batch_norm=False, layer_norm=True)
        self.node_mlp = MLP(mlp_2_in_dim, hidden_dim_2, 768//2,
                            2, 0.2, batch_norm=False, layer_norm=True)

    def forward(self, x, edge_index, edge_attr, u, node_batch, edge_batch, wts):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        # import ipdb
        # ipdb.set_trace()
        row, col = edge_index
        edge_message = torch.cat([x[row], edge_attr], dim=1)
        edge_message = self.message_mlp(edge_message)  # edge_message: [#edges, hidden_dim]
        if wts is None:
            received_message = scatter_mean(edge_message, col, dim=0, dim_size=x.size(0))
        else:
            received_message = scatter_mean(edge_message * wts, col, dim=0, dim_size=x.size(0))
        if "no_vec" not in self.ablation:
            out = torch.cat([x, received_message, u[node_batch]], dim=1)
        else:
            out = torch.cat([x, received_message], dim=1)
        return self.node_mlp(out)


class GraphNetwork(torch.nn.Module):
    def __init__(self, ablation, u_dim):
        super(GraphNetwork, self).__init__()
        self.ablation = ablation
        # self.edge_model = edge_model
        # self.node_model = node_model
        self.u_dim = u_dim
        lm = AutoModelForSequenceClassification.from_pretrained("roberta-base")
        self.input_embeddings = lm.get_input_embeddings()
        embedding_dim = self.input_embeddings.embedding_dim
        self.node_model = NodeModel(embedding_dim, 768//2, 128, 128, u_dim, ablation)
        self.edge_model = EdgeModel(embedding_dim, 128, 128, 0.1, u_dim, ablation)
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    def forward(self, x, edge_index, edge_attr, u, node_batch):
        # import ipdb
        # ipdb.set_trace()
        x = x[:, 1:]          #remove cls token
        node_attr = self.input_embeddings(x)  #calculate embedddings for nodes
        node_mask = 1 - ((x == self.tokenizer.pad_token_id) +( x == self.tokenizer.sep_token_id)).float().unsqueeze(2)     #calculate node attention_mask
        node_attr = (node_attr * node_mask).mean(1)                           #calculate mean emb for all tokens of nodes
        x = node_attr
        edge_attr = edge_attr[:, 1:]                                     #remove cls token
        edge_mask = 1 - ((edge_attr == self.tokenizer.pad_token_id) + ( edge_attr == self.tokenizer.sep_token_id)).float().unsqueeze(2) #calcualte edge attention mask
        edge_attr = self.input_embeddings(edge_attr)                     #calculate embeddings for edges
        edge_attr = (edge_attr*edge_mask).mean(1)                             #calculate mean embs for all edge tokens
        row, col = edge_index
        edge_batch = node_batch[row]
        edge_attr, unnormalized_wts, normalized_wts = self.edge_model(x[row], x[col], edge_attr, u, edge_batch)
        unnormalized_wts = torch.sigmoid(unnormalized_wts)
        if 'no_edge_weight' in self.ablation:
            x = self.node_model(x, edge_index, edge_attr, u, node_batch, edge_batch, None)
        else:
            if 'unnormalized_edge_weight' in self.ablation:
                x = self.node_model(x, edge_index, edge_attr, u, node_batch, edge_batch, unnormalized_wts)
            else:
                x = self.node_model(x, edge_index, edge_attr, u, node_batch, edge_batch, normalized_wts)
        pooled_edge_vecs = scatter_mean(edge_attr * normalized_wts, edge_batch, dim=0, dim_size=u.shape[0])
        pooled_node_vecs = scatter_mean(x, node_batch, dim=0, dim_size=u.shape[0])
        graph_vecs = torch.cat([pooled_edge_vecs, pooled_node_vecs], dim = 1)
        return graph_vecs, u, normalized_wts

class MultimodalGN(torch.nn.Module):
    def __init__(self, ablation, alpha):
        super(MultimodalGN, self).__init__()
        self.ablation = ablation
        self.graph_network = GraphNetwork(ablation, 768)
        self.alpha = alpha
        self.classifier = MLP(768, 128, 2, 2, 0.1, batch_norm=False, layer_norm=True)
        self.loss_func = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, graph_batch, u, labels):
        u = u.cuda()
        x = graph_batch.x
        x = x.cuda()
        edge_index = graph_batch.edge_index
        edge_index = edge_index.cuda()
        edge_attr = graph_batch.edge_attr
        edge_attr = edge_attr.cuda()
        node_batch = graph_batch.batch
        node_batch = node_batch.cuda()
        labels = labels.cuda()
        graph_vecs, u, normalized_wts = self.graph_network(x, edge_index, edge_attr, u, node_batch)
        # import ipdb
        # ipdb.set_trace()
        final_vecs = self.alpha*u + (1-self.alpha)*graph_vecs
        logits = self.classifier(final_vecs)
        loss = self.loss_func(logits, labels)
        return loss, logits, normalized_wts


def train(model, train_loader, eval_loader, optimizer, epochs, scheduler, ablation, loss_alpha, is_train=True):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total = 1
        if is_train:
            for batch in tqdm(train_loader):
                # import ipdb
                # ipdb.set_trace()
                graph_batch = batch["graph"]
                u = batch["u"]
                row, col = batch["graph"].edge_index
                node_batch = batch["graph"].batch
                loss, logits, normalized_wts = model(graph_batch, u, batch["label"])
                labels = batch["label"].cuda()
                total_correct = total_correct + (logits.argmax(1) == labels).sum().item()
                total = total + batch["label"].shape[0]
                if 'no_edge_weight' not in ablation:
                    log_wts = torch.log(normalized_wts + 0.0000001)
                    entropy = - normalized_wts * log_wts  # entropy: [num_of_edges in the batched graph, 1]
                    entropy = entropy.cuda()
                    node_batch = node_batch.cuda()
                    entropy = scatter_mean(entropy, node_batch[row], dim=0, dim_size=batch["u"].shape[0])
                    loss += loss_alpha * torch.mean(entropy)
                loss.backward() 
                total_loss += loss.item()
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()  # bp: scale: loss per question
                scheduler.step()
                optimizer.zero_grad()
        model.eval()
        total_correct_eval = 0
        total_eval = 0
        correct_ans = {}
        for batch in tqdm(eval_loader):
            # import ipdb
            # ipdb.set_trace()
            graph_batch = batch["graph"]
            u = batch["u"]
            row, col = batch["graph"].edge_index
            node_batch = batch["graph"].batch
            loss, logits, _ = model(graph_batch, u, batch["label"])
            labels = batch["label"].cuda()
            total_correct_eval = total_correct_eval + (logits.argmax(1) == labels).sum().item()
            total_eval = total_eval + batch["label"].shape[0]
            for i in range(batch["label"].shape[0]):
                if logits[i][1] > logits[i][0]:
                    correct_ans[batch["identifier"][i]] = 1
                else:
                    correct_ans[batch["identifier"][i]] = 0
        # print(total_correct_eval/total_eval)
        print("Epoch: {}, Loss: {}, Train Accuracy: {}, Eval Accuracy: {}".format(epoch, total_loss/total, total_correct/total, total_correct_eval/total_eval))
        return total_correct_eval/total_eval, correct_ans


train_dataset = SceneGraphDataset(data_dir='/data/mrigankr/mml/nlvr2/data/train.json', graph_dir="scene_graph_benchmark/nlvr2_train_100.json", image_dir="/data/mrigankr/mml/train/", vecs_dir = "/data//mrigankr/pgm/train_fts_xvlm/", transform=None, pre_transform=None, pre_filter=None)
dev_dataset = SceneGraphDataset(data_dir='/data/mrigankr/mml/nlvr2/data/dev.json', graph_dir="scene_graph_benchmark/nlvr2_dev_100.json", image_dir="/data/mrigankr/mml/dev/", vecs_dir = "/data//mrigankr/pgm/dev_fts_xvlm/", transform=None, pre_transform=None, pre_filter=None)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
dev_loader = DataLoader(dev_dataset, batch_size=8, shuffle=True, num_workers=4)
ablation = ["edge_weight", "use_vec"]
alpha = 0.8
loss_alpha = 0.6
model = MultimodalGN(ablation, alpha)
model = model.cuda()
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01, 'lr': 1e-3},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': 1e-3},
    # {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.decoder_lr},
    # {'params': [p for n, p in model.decoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.decoder_lr},
]
optimizer = torch.optim.Adam(grouped_parameters, lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
# model.load_state_dict(torch.load("ours_model.pt"))
_, correct_model1 = train(model, train_loader, dev_loader, optimizer, 1, scheduler, ablation, loss_alpha, is_train = True)
#save the model
torch.save(model.state_dict(), "model.pt")
