from utils import *
from deeprobust.graph.defense.noisy_gcn import Noisy_GCN
import torch.nn.functional as F
from GCORN import GCORN
import pandas as pd
import argparse


def get_graph_representations(data, device):
    """根据data对象生成邻接矩阵的稀疏张量或归一化版本。"""
    adj_sp = torch.sparse_coo_tensor(
        data.edge_index,
        torch.ones(data.edge_index.size(1), device=device),
        size=(data.num_nodes, data.num_nodes)
    ).to(device)
    norm_adj = None
    if 'normalize_tensor_adj_from_edge_index' in globals():
        norm_adj = normalize_tensor_adj_from_edge_index(data.edge_index, data.num_nodes).to(device)
    return adj_sp, norm_adj


def train_single_model_on_clean(model_name, clean_data, labels, idx_train, idx_val, device, args, logger):
    """
    在干净图上训练【单次】模型，并返回训练好的模型及其在干净图上的预测。
    """
    n_features = clean_data.x.size(1)
    n_classes = labels.max().item() + 1
    model = None

    # --- 模型初始化和训练 ---
    if model_name in ['gcn', 'gnn-guard', 'rgcn', 'gcn-jaccard']:
        model = load_pyg_model(clean_data, model_name, args.source, args.dataset, device, logger, True)
        model.fit(clean_data, verbose=False, train_iters=300, patience=400)
    elif model_name == 'gcorn':
        model = GCORN(n_features, 16, n_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model = train_gcorn(model, optimizer, clean_data)
    elif model_name == 'noisy':
        best_beta_acc_val = 0
        best_beta_model = None
        adj_sp, _ = get_graph_representations(clean_data, device)
        for beta in np.arange(0, 0.15, 0.01):
            temp_model = Noisy_GCN(nfeat=n_features, nhid=16, nclass=n_classes, dropout=0.5, device=device,
                                   noise_ratio_1=beta).to(device)
            temp_model.fit(clean_data.x, adj_sp, clean_data.y, idx_train, train_iters=200, idx_val=idx_val,
                           verbose=False)
            temp_model.eval()
            acc_val, _ = temp_model.test(idx_val)
            if acc_val > best_beta_acc_val:
                best_beta_acc_val = acc_val
                best_beta_model = temp_model
        model = best_beta_model

    model.eval()

    # --- 获取模型在干净图上的预测 ---
    with torch.no_grad():
        if model_name in ['gcn', 'gnn-guard', 'rgcn', 'gcn-jaccard']:
            preds = model.predict(clean_data.x, clean_data.adj_t)
        elif model_name == 'gcorn':
            _, norm_adj = get_graph_representations(clean_data, device)
            preds = model(clean_data.x, norm_adj)
        elif model_name == 'noisy':
            adj_sp, _ = get_graph_representations(clean_data, device)
            preds = model(clean_data.x, adj_sp)

    return model, preds


def evaluate_model(model, model_name, data, labels, idx_test, device):
    """使用预训练的模型在给定的图上进行评估。"""
    model.eval()
    with torch.no_grad():
        if model_name in ['gcn', 'gnn-guard', 'rgcn', 'gcn-jaccard']:
            preds = model.predict(data.x, data.adj_t)
        elif model_name == 'gcorn':
            _, norm_adj = get_graph_representations(data, device)
            preds = model(data.x, norm_adj)
        elif model_name == 'noisy':
            adj_sp, _ = get_graph_representations(data, device)
            preds = model(data.x, adj_sp)

    test_acc = calculate_accuracy(idx_test, preds.argmax(1), labels)
    return test_acc


def replace_original_graph_with_mod_adj(iter_pyg_data, pyg_data, mod_adj):
    """
    用被攻击的邻接矩阵替换iter_pyg_data中的原始图部分，但保留与新注入节点相关的边。
    """
    device = iter_pyg_data.x.device
    num_orig_nodes = pyg_data.x.size(0)

    mod_edge_index, _ = from_scipy_sparse_matrix(mod_adj)
    mod_edge_index = mod_edge_index.to(device)
    mod_edge_index = to_undirected(mod_edge_index)

    old_edge_index = iter_pyg_data.edge_index
    mask_inserted_edges = (
            (old_edge_index[0] >= num_orig_nodes) |
            (old_edge_index[1] >= num_orig_nodes)
    )
    inserted_edge_index = old_edge_index[:, mask_inserted_edges]

    new_edge_index = torch.cat([mod_edge_index, inserted_edge_index], dim=1)

    # 直接在副本上操作，避免修改原始数据
    final_graph = iter_pyg_data.clone()
    final_graph.edge_index = new_edge_index
    final_graph = ToSparseTensor(remove_edge_index=False)(final_graph)

    return final_graph


def run_evasion_experiment(args, device):
    """执行完整的逃逸攻击实验流程"""
    # --- 1. 数据加载和预处理 (只需一次) ---
    logger = get_logger('models.log', level=0)
    pyg_data = get_data(args.dataset).to(device)
    mod_adj = load_attack_adj(args.attack_name, args.dataset, args.rate)

    mod_pyg_data = pyg_data.clone()
    mod_pyg_data.adj_t = retype_adj(mod_adj)
    coo_adj = mod_adj.tocoo()
    mod_pyg_data.edge_index = torch.tensor(np.vstack((coo_adj.row, coo_adj.col)), dtype=torch.long, device=device)

    labels = pyg_data.y.cpu().numpy()
    idx_train = torch.nonzero(pyg_data.train_mask).squeeze()
    idx_test = torch.nonzero(pyg_data.test_mask).squeeze()
    idx_val = torch.nonzero(pyg_data.val_mask).squeeze()

    # --- 2. 重复执行n次完整的实验 ---
    clean_accs = []
    evasion_before_accs = []
    evasion_after_accs = []

    for i in range(args.repeat_n):
        print(f"\n--- Starting Run {i + 1}/{args.repeat_n} ---")

        # a. 训练一个新模型
        trained_model, preds_on_clean_graph = train_single_model_on_clean(
            args.model_name, pyg_data, labels, idx_train, idx_val, device, args, logger
        )

        # b. 在干净图上评估
        clean_acc = evaluate_model(trained_model, args.model_name, pyg_data, labels, idx_test, device)
        clean_accs.append(clean_acc)
        print(f"Run {i + 1} Accuracy on Clean Graph: {clean_acc:.4f}")

        # c. 在被攻击图上评估（防御前）
        evasion_acc_before = evaluate_model(trained_model, args.model_name, mod_pyg_data, labels, idx_test, device)
        evasion_before_accs.append(evasion_acc_before)
        print(f"Run {i + 1} Evasion Accuracy (Before Defense): {evasion_acc_before:.4f}")

        # d. 执行防御并评估（防御后）
        if args.add_node_n > 0:
            iter_pyg_data = Initial_inject_nodes(mod_pyg_data, n=args.add_node_n, k=args.mean_node_n).to(device)
            injected_node_ids = torch.arange(pyg_data.num_nodes, iter_pyg_data.num_nodes, device=device)
            all_node_idx = torch.arange(pyg_data.num_nodes, device=device)
            nodes_to_connect = all_node_idx[~torch.isin(all_node_idx, idx_train)]

            iter_pyg_data, _ = connect_topk_confident_nodes_once(
                iter_pyg_data, preds_on_clean_graph, injected_node_ids, nodes_to_connect, top_k=len(nodes_to_connect)
            )

            final_evasion_graph = replace_original_graph_with_mod_adj(iter_pyg_data, pyg_data, mod_adj)

            trained_model, preds_on_clean_graph = train_single_model_on_clean(
                args.model_name, final_evasion_graph, labels, idx_train, idx_val, device, args, logger
            )
            evasion_acc_after = evaluate_model(trained_model, args.model_name, final_evasion_graph, labels, idx_test,
                                               device)
            evasion_after_accs.append(evasion_acc_after)
            print(f"Run {i + 1} Evasion Accuracy (After Defense): {evasion_acc_after:.4f}")

    # --- 3. 计算最终结果的均值和方差 ---
    result = []
    clean_mean, clean_std = compute_mean_std(clean_accs)
    result.append(['clean_graph', clean_mean, clean_std])
    print(f"\nAverage Clean Graph Accuracy: {clean_mean:.4f} ± {clean_std:.4f}")

    before_mean, before_std = compute_mean_std(evasion_before_accs)
    result.append(['evasion_before_defense', before_mean, before_std])
    print(f"Average Evasion Accuracy (Before Defense): {before_mean:.4f} ± {before_std:.4f}")

    if args.add_node_n > 0:
        after_mean, after_std = compute_mean_std(evasion_after_accs)
        result.append(['evasion_after_defense', after_mean, after_std])
        print(f"Average Evasion Accuracy (After Defense): {after_mean:.4f} ± {after_std:.4f}")

    # --- 4. 保存结果 ---
    output_dir = "result_evasion"
    os.makedirs(output_dir, exist_ok=True)
    file_name = (
        f"{output_dir}/{args.dataset}_{args.attack_name}_{args.model_name}_"
        f"rate{args.rate}_add{args.add_node_n}.csv"
    )
    df = pd.DataFrame(result, columns=["Stage", "Mean Accuracy", "Std"])
    df["Mean Accuracy"] = (df["Mean Accuracy"] * 100).round(2)
    df["Std"] = (df["Std"] * 100).round(2)
    df.to_csv(file_name, index=False)
    print(f"\nEvasion results saved to {file_name}")
    print(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph Evasion Attack and Defense')
    parser.add_argument('--mode', type=str, default='evasion', choices=['evasion'], help='Set experiment mode')
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset name')
    parser.add_argument('--model_name', type=str, default='gcn',
                        choices=['gcn', 'gnn-guard', 'rgcn', 'gcn-jaccard', 'noisy', 'gcorn'], help='Model name')
    parser.add_argument('--source', type=str, default='gcn', help='Source model for attack')
    parser.add_argument('--attack_name', type=str, default='prbcd',
                        choices=['greedy', 'prbcd', 'pga', 'pgdattack-CW', 'Metattack', 'MinMax', 'DICE'],
                        help='Attack method')
    parser.add_argument('--rate', type=float, default=0.25, help='Attack perturbation rate')
    parser.add_argument('--add_node_n', type=int, default=5, help='Number of nodes to add for defense')
    parser.add_argument('--mean_node_n', type=int, default=5,
                        help='Mean number of connections for each new node during initialization')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--repeat_n', type=int, default=4, help='Number of repetitions for training the clean model')

    args = parser.parse_args()
    device = get_device(args.gpu_id)

    if args.mode == 'evasion':
        run_evasion_experiment(args, device)
