import scipy.sparse as sp
import os
import torch
from scipy.sparse import csr_matrix
import numpy as np
from deeprobust.graph import utils as _utils
from torch_geometric.datasets import Planetoid
from deeprobust.graph.data import Dataset, Dpr2Pyg
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
from torch_sparse import SparseTensor
import random
from models import model_map, choice_map
import logging
from torch_geometric.utils import from_scipy_sparse_matrix, to_undirected
from torch_geometric.transforms import ToSparseTensor
import torch.nn.functional as F
import copy
def compute_mean_std(lst):
    arr = np.array(lst)
    return arr.mean(), arr.std()

def get_logger(filename, level=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[level])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger
def calculate_accuracy(idx_test, preds, labels):
    """
    计算并返回原始准确率和攻击后准确率。

    参数:
    idx_test (list): 测试集索引
    preds (list): 攻击前的预测结果
    labels (list): 实际标签
    返回:
    ori_acc (float): 攻击前准确率
    """
    correct = sum([1 for idx in idx_test if preds[idx] == labels[idx]])
    ori_acc = correct / (len(idx_test) + 0.00000001)
    return ori_acc

def normalize_tensor_adj_from_edge_index(edge_index, num_nodes):
    """
    接受 edge_index 格式的邻接信息，返回 dense tensor 格式的 symmetric normalized adjacency matrix。
    输入:
        - edge_index: torch.LongTensor, shape [2, num_edges]
        - num_nodes: int
    输出:
        - dense 对称归一化邻接矩阵: torch.FloatTensor, shape [num_nodes, num_nodes]
    """
    device = edge_index.device

    # 创建稀疏邻接矩阵，所有边权为 1
    edge_weight = torch.ones(edge_index.size(1), device=device)
    adj = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes), device=device)

    # 转为 dense 并加自环
    A = adj.to_dense()
    A = A + torch.eye(num_nodes, device=device)

    # 计算度矩阵的 power 次
    D_power = A.sum(1).pow(-0.5)
    D_power[torch.isinf(D_power)] = 0.
    D_mat = torch.diag(D_power)

    return D_mat @ A @ D_mat

def retype_adj(mod_adj):
    adj_coo = mod_adj.tocoo()
    row = torch.tensor(adj_coo.row, dtype=torch.long)
    col = torch.tensor(adj_coo.col, dtype=torch.long)
    value = torch.tensor(adj_coo.data, dtype=torch.float32)
    return SparseTensor(row=row, col=col, value=value, sparse_sizes=mod_adj.shape)
def get_device(gpu_id):
    if torch.cuda.is_available() and gpu_id >= 0:
        device = f'cuda:{gpu_id}'
    else:
        device = 'cpu'
    return device
def normalize_feature_tensor(x):
    x = _utils.to_scipy(x)
    x = _utils.normalize_feature(x)
    x = torch.FloatTensor(np.array(x.todense()))
    return x
def get_data(name = 'cora', source='gcn'):
    path = './dataset'
    dataset = Dataset(root=path, name=name, setting=source, seed=15)
    dataset = Dpr2Pyg(dataset, transform=T.ToSparseTensor(remove_edge_index=False))
    pyg_data = dataset[0]
    pyg_data.num_classes = dataset.num_classes
    pyg_data.x = normalize_feature_tensor(pyg_data.x)
    return pyg_data

def load_attack_adj(attack_method,dataset,ptb_rate):

    # if source=='prognn' and attack_method=='pga' and dataset=='cora':
    #     attack_path = '../PGA-main/attack/perturbed_adjs'
    attack_path = './attack_data'
    if attack_method in ['greedy','prbcd','pga','pgdattack-CW']:
        filename = attack_method + '-' + dataset + '-' + f'{ptb_rate}' + '.pth'
        filename = os.path.join(attack_path, filename)
        data = torch.load(filename)
        modified_adj_list = data['modified_adj_list']
        # attack_config = data['attack_config']
        mod_adj = modified_adj_list[0]
        # 获取 COO 格式的行、列索引和非零值
        row, col, values = mod_adj.coo()
        if values is None:
            values = torch.ones(row.size(0))  # 生成与非零元素个数一致的权重
        # 获取 shape 信息
        num_rows, num_cols = mod_adj.sizes()

        # 转换为 CSR 格式
        csr_mod_adj = csr_matrix((values.numpy(), (row.cpu().numpy(), col.cpu().numpy())),
                                 shape=(num_rows, num_cols))

        # 将 csr_mod_adj 转换为 numpy 数组
        csr_mod_adj_np = csr_mod_adj
        # csr_mod_adj_np = csr_mod_adj.toarray()  # 如果 csr_mod_adj 是 CSR 格式
    elif attack_method in ['Metattack', 'MinMax', 'PGDAttack', 'DICE', 'DICE_train',]:
        try:
            filename = attack_method + '-' + dataset + '-' + f'{ptb_rate}' + '.npz'
            filename = os.path.join(attack_path, filename)
            csr_mod_adj_np = sp.load_npz(filename)
        except:
            filename = attack_method + '-' + dataset + '-' + f'{ptb_rate}' + '.npy'
            filename = os.path.join(attack_path, filename)
            csr_mod_adj_np = np.load(filename)
    return csr_mod_adj_np

def load_pyg_model(pyg_data,model_name,source,dataset,device,logger,pretrained):
    # model_name = args.victim
    save_path = f"./victims/"
    save_path += f"{model_name + '-' + dataset}.pth"
    savings = torch.load(save_path)
    config = savings['config']
    states = savings['state_dicts']
    performance = savings['performance']
    model_func = model_map[model_name]
    model = model_func(config=config, pyg_data=pyg_data, device=device, logger=logger)
    model = model.to(device)
    model.load_state_dict(states[0])
    if pretrained == False:
        model.fit(pyg_data)
    return model

def Initial_inject_nodes(pyg_data, n=5, k=5, device='cuda'):
    pyg_data = pyg_data.clone()
    features = pyg_data.x.to(device)
    labels = pyg_data.y.to(device)
    edge_index = pyg_data.edge_index.to(device)
    train_mask = pyg_data.train_mask.to(device)
    num_classes = pyg_data.num_classes
    num_nodes = features.size(0)
    feat_dim = features.size(1)

    new_features = []
    new_labels = []
    new_edges = []

    for cls in range(num_classes):
        cls_indices = torch.nonzero(train_mask & (labels == cls)).squeeze()
        if cls_indices.numel() == 0:
            continue
        for _ in range(n):
            sampled = cls_indices[torch.randperm(cls_indices.size(0))[:k]]
            mean_feat = features[sampled].mean(dim=0)
            new_features.append(mean_feat)
            new_labels.append(cls)

            new_node_idx = num_nodes + len(new_features) - 1

            # 连接新节点到所有训练节点
            edge_src = torch.full((cls_indices.size(0),), new_node_idx, dtype=torch.long, device=device)
            new_edges.append(torch.stack([edge_src, cls_indices]))  # 新节点到训练节点的边
            new_edges.append(torch.stack([cls_indices, edge_src]))  # 训练节点到新节点的边

    if len(new_features) > 0:
        new_features = torch.stack(new_features)
        new_labels = torch.tensor(new_labels, dtype=torch.long, device=device)
        new_edges = torch.cat(new_edges, dim=1)

        pyg_data.x = torch.cat([features, new_features], dim=0)
        pyg_data.y = torch.cat([labels, new_labels], dim=0).to(device)
        pyg_data.edge_index = torch.cat([edge_index, new_edges], dim=1)

        # 更新 train_mask
        new_train_mask = torch.zeros(pyg_data.y.shape[0], dtype=torch.bool, device=device)
        new_train_mask[:num_nodes] = train_mask
        new_train_mask[num_nodes:] = True
        pyg_data.train_mask = new_train_mask

        # 扩展 val_mask 和 test_mask（新增节点不参与）
        for mask_name in ['val_mask', 'test_mask']:
            if hasattr(pyg_data, mask_name):
                old_mask = getattr(pyg_data, mask_name).to(device)
                new_mask = torch.zeros(pyg_data.y.size(0), dtype=torch.bool, device=device)
                new_mask[:old_mask.size(0)] = old_mask
                setattr(pyg_data, mask_name, new_mask)

        # 更新 adj_t（PyG 的 SparseTensor）
        new_adj = SparseTensor.from_edge_index(
            pyg_data.edge_index,
            sparse_sizes=(pyg_data.x.size(0), pyg_data.x.size(0))
        ).coalesce()
        pyg_data.adj_t = new_adj

    return pyg_data

def connect_topk_confident_nodes_once(
        pyg_data,
        preds,
        injected_node_ids,
        not_connected_node_idx,
        top_k=50,
):
    """
    一轮连接置信度最高的同类节点。

    参数:
    - pyg_data: 当前图数据
    - model: 已训练模型
    - injected_node_ids: tensor, 插入的节点索引
    - not_connected_node_idx: tensor, 当前未连接的节点索引
    - top_k: 每轮连接的节点数

    返回:
    - iter_pyg_data: 新图数据，更新了 edge_index 和 adj_t
    - updated_not_connected_idx: 移除已连接后的新未连接节点索引
    """

    device = pyg_data.x.device
    probs = torch.softmax(preds, dim=1)

    # 分类边界差值（最高概率 - 第二高概率）
    top2 = torch.topk(probs[not_connected_node_idx], 2, dim=1)
    confidence_margin = top2.values[:, 0] - top2.values[:, 1]

    # 取置信度差值最大的top_k个点
    top_conf_idx = torch.topk(confidence_margin, min(top_k, len(not_connected_node_idx))).indices
    selected_nodes = not_connected_node_idx[top_conf_idx]

    # 预测的类别标签
    pred_classes = probs[selected_nodes].argmax(dim=1)

    device = injected_node_ids.device
    injected_node_labels = pyg_data.y.to(device)[injected_node_ids]

    # 构造边（双向）
    new_edges = []
    for i in range(len(selected_nodes)):
        pred_cls = pred_classes[i].item()
        src_node = selected_nodes[i].item()

        # 找到插入节点中对应类别的节点
        target_nodes = injected_node_ids[injected_node_labels == pred_cls]

        if len(target_nodes) == 0:
            continue  # 无可连目标

        edges_src = torch.full((len(target_nodes),), src_node, device=device)
        edges_dst = target_nodes

        new_edges.append(torch.stack([edges_src, edges_dst], dim=0))
        new_edges.append(torch.stack([edges_dst, edges_src], dim=0))  # 双向

    if new_edges:
        new_edges_tensor = torch.cat(new_edges, dim=1)
        edge_index = torch.cat([pyg_data.edge_index, new_edges_tensor], dim=1)
    else:
        edge_index = pyg_data.edge_index.clone()

    # 构建新的 SparseTensor adj_t
    adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(pyg_data.num_nodes, pyg_data.num_nodes))
    adj_t = adj_t.to_symmetric()

    # 更新 pyg_data
    iter_pyg_data = pyg_data.clone()
    iter_pyg_data.edge_index = edge_index
    iter_pyg_data.adj_t = adj_t

    # 更新未连接节点索引
    updated_not_connected_idx = not_connected_node_idx[~torch.isin(not_connected_node_idx, selected_nodes)]

    return iter_pyg_data, updated_not_connected_idx


def train_gcorn(model_ro, optimizer, pyg_data, epochs=300):
    device = pyg_data.x.device
    idx_val = torch.nonzero(pyg_data.val_mask).squeeze()
    labels = pyg_data.y.cpu().numpy()
    norm_adj = normalize_tensor_adj_from_edge_index(pyg_data.edge_index, pyg_data.x.size(0)).to(device)
    best_val_acc = 0
    for epoch in range(epochs):
        model_ro.train()
        optimizer.zero_grad()
        out = model_ro(pyg_data.x, norm_adj)
        loss = F.cross_entropy(out[pyg_data.train_mask],
                               pyg_data.y[pyg_data.train_mask])
        loss.backward()
        optimizer.step()
        preds_ori = model_ro(pyg_data.x, norm_adj)
        preds_label = preds_ori.argmax(1)
        val_acc = calculate_accuracy(idx_val, preds_label, labels)
        if val_acc > best_val_acc:
            best_model_ro = copy.deepcopy(model_ro)
            best_val_acc = val_acc

    return best_model_ro