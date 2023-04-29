from gcn_dataloader import SceneGraphDataset
from torch_geometric.loader import DataLoader


dataset = SceneGraphDataset(data_dir='/data/mrigankr/mml/nlvr2/data/dev.json', graph_dir="scene_graph_benchmark/nlvr2_dev_100.json", image_dir="/data/mrigankr/mml/dev/", transform=None, pre_transform=None, pre_filter=None)

loader = DataLoader(dataset, batch_size=8, shuffle=True)

for batch in loader:
    import ipdb
    ipdb.set_trace()

