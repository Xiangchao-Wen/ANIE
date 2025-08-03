from utils import *
from deeprobust.graph.defense.noisy_gcn import Noisy_GCN
import torch.nn.functional as F
from GCORN import GCORN
import pandas as pd
import argparse
import time
from torch_geometric.utils import homophily
from scipy.stats import ttest_ind # 导入独立样本T检验函数
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import ks_2samp # 导入K-S检验函数
import logging
from scipy.stats import gaussian_kde, ttest_ind, ks_2samp, wilcoxon # 增加了 ks_2samp, wilcoxon
from scipy.spatial.distance import jensenshannon # 增加了 jensenshannon
import json

# 将 matplotlib 的日志级别设置为 WARNING，可以有效过滤掉 DEBUG 和 INFO 级别的日志
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)
# 1. 获取根日志记录器 (所有日志的源头)
root_logger = logging.getLogger()

# 2. 设置根记录器的级别为 WARNING
#    这会确保只有 WARNING 及以上级别的日志消息才能通过这个记录器
root_logger.setLevel(logging.WARNING)

# 3. 遍历所有现有的处理器（Handler），并将它们的级别也设为 WARNING
#    这是关键一步，因为处理器也有自己的级别过滤器
for handler in root_logger.handlers:
    handler.setLevel(logging.WARNING)

# 设置字体为 Times New Roman
mpl.rcParams['font.family'] = 'Times New Roman'

# 设置嵌入 TrueType 字体（PDF/EPS 可复制、高清缩放）
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 14,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 14,
})
from scipy.stats import chisquare


def run_chi_squared_analysis(scores_dict):
    """
    使用卡方拟合优度检验来比较分布。

    Args:
        scores_dict (dict): 包含 'Ori', 'After Attack', 'After RNE' 得分的字典。
    """
    if 'Ori' not in scores_dict:
        print("错误：scores_dict 中缺少 'Ori' 数据。")
        return

    # --- 1. 数据准备 (分箱) ---
    # 定义用于所有分布的箱体边界，这是保证比较公平的关键
    num_bins = 40  # 您可以根据数据调整箱体数量
    bins = np.linspace(0, 1, num_bins + 1)

    # 获取 'Ori' 分布的频数，作为我们的“期望”基准
    scores_ori = scores_dict['Ori']
    # 使用 np.histogram 计算每个箱体内的样本数量（频数）
    expected_counts, _ = np.histogram(scores_ori, bins=bins)

    # --- 注意事项：处理频数为0的箱体 ---
    # 卡方检验中，期望频数不能为0。如果为0，我们用一个很小的值代替。
    # 另外，如果期望频数过小（通常建议>5），检验的准确性会下降。
    if np.any(expected_counts < 5):
        print("警告：期望频数中有小于5的箱体，卡方检验的准确性可能受影响。")
    expected_counts = np.where(expected_counts == 0, 1e-10, expected_counts)

    print("\n--- Chi-squared Test Results ---")
    print("原假设 (H0): 观测分布与期望分布(Ori)一致。")
    print("我们期望 Attack 的 p-value 小，RNE 的 p-value 相对更大。")
    print("-" * 30)

    # --- 2. 检验 'After Attack' 分布 ---
    if 'After Attack' in scores_dict:
        scores_att = scores_dict['After Attack']
        observed_counts_att, _ = np.histogram(scores_att, bins=bins)

        # 为了比较，需要将期望频数的总和调整为与观测频数总和相同
        sum_ori = np.sum(expected_counts)
        sum_att = np.sum(observed_counts_att)
        scaled_expected_counts = expected_counts * (sum_att / sum_ori)
        # 再次检查并替换0值
        scaled_expected_counts = np.where(scaled_expected_counts == 0, 1e-10, scaled_expected_counts)

        chi2_stat, p_val = chisquare(f_obs=observed_counts_att, f_exp=scaled_expected_counts)

        print(f"Ori vs. After Attack:")
        print(f"  Chi-squared Statistic = {chi2_stat:.4f}")
        print(f"  p-value = {p_val:.3g}")
        if p_val < 0.05:
            print("  结论: 攻击显著改变了原始分布 (p < 0.05)。")
        else:
            print("  结论: 攻击未显著改变原始分布 (p >= 0.05)。")
        print("-" * 30)

    # --- 3. 检验 'After RNE' 分布 ---
    if 'After RNE' in scores_dict:
        scores_rne = scores_dict['After RNE']
        observed_counts_rne, _ = np.histogram(scores_rne, bins=bins)

        sum_rne = np.sum(observed_counts_rne)
        scaled_expected_counts = expected_counts * (sum_rne / sum_ori)
        scaled_expected_counts = np.where(scaled_expected_counts == 0, 1e-10, scaled_expected_counts)

        chi2_stat, p_val = chisquare(f_obs=observed_counts_rne, f_exp=scaled_expected_counts)

        print(f"Ori vs. After RNE:")
        print(f"  Chi-squared Statistic = {chi2_stat:.4f}")
        print(f"  p-value = {p_val:.3g}")
        if p_val < 0.05:
            print("  结论: 免疫后的分布仍与原始分布有显著差异 (p < 0.05)。")
        else:
            print("  结论: 免疫后的分布与原始分布无显著差异 (p >= 0.05)。")
        print("-" * 30)


def calculate_graph_homophily(data: 'torch_geometric.data.Data') -> float:
    """
    计算给定 PyG 数据对象的全图同质率。

    参数:
        data (torch_geometric.data.Data): 包含图信息的 PyG 数据对象。
                                           必须包含 data.edge_index 和 data.y。

    返回:
        float: 图的同质率。
    """
    print("开始计算同质率...")
    # 从数据对象中提取边索引和节点标签
    edge_index = data.edge_index
    labels = data.y

    # 调用 PyG 内置的 homophily 函数
    # method='edge' 是计算同质率的默认和标准方法
    h = homophily(edge_index, labels, method='edge')

    # .item() 用于从单元素张量中提取 Python 数值
    return h
def get_graph_representations(data, device):
    """根据data对象生成邻接矩阵的稀疏张量或归一化版本。"""
    adj_sp = torch.sparse_coo_tensor(
        data.edge_index,
        torch.ones(data.edge_index.size(1), device=device),
        size=(data.num_nodes, data.num_nodes)
    ).to(device)
    # gcorn 需要归一化的邻接矩阵
    norm_adj = None
    if 'normalize_tensor_adj_from_edge_index' in globals():  # 检查函数是否存在
        norm_adj = normalize_tensor_adj_from_edge_index(data.edge_index, data.num_nodes).to(device)
    return adj_sp, norm_adj


def run_single_trial(model_name, data, labels, idx_train, idx_val, idx_test, device, args, logger):
    """
    运行单次实验：初始化、训练并评估一个模型。
    返回测试集准确率和模型在整个图上的预测。
    """
    # 1. 初始化模型
    n_features = data.x.size(1)
    n_classes = labels.max().item() + 1

    if model_name in ['gcn', 'gnn-guard', 'rgcn', 'gcn-jaccard']:
        model = load_pyg_model(data, model_name, args.source, args.dataset, device, logger, True)
        model.fit(data, verbose=False, train_iters=300, patience=400)
        with torch.no_grad():
            preds = model.predict(data.x, data.adj_t)

    elif model_name == 'gcorn':
        model = GCORN(n_features, 16, n_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        # 假设 train_gcorn 返回训练好的模型
        model = train_gcorn(model, optimizer, data)
        model.eval()
        _, norm_adj = get_graph_representations(data, device)
        with torch.no_grad():
            preds = model(data.x, norm_adj)

    elif model_name == 'noisy':
        best_acc_val = 0
        best_model = None
        adj_sp, _ = get_graph_representations(data, device)

        # 为NoisyGCN寻找最佳的beta超参数
        for beta in np.arange(0, 0.15, 0.01):
            model = Noisy_GCN(nfeat=n_features, nhid=16, nclass=n_classes, dropout=0.5, device=device,
                              noise_ratio_1=beta).to(device)
            model.fit(data.x, adj_sp, data.y, idx_train, train_iters=200, idx_val=idx_val, verbose=False)
            model.eval()
            acc_val, _ = model.test(idx_val)

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                best_model = model

        model = best_model
        with torch.no_grad():
            preds = model(data.x, adj_sp)

    # 2. 计算准确率
    test_acc = calculate_accuracy(idx_test, preds.argmax(1), labels)
    return test_acc, preds


def run_repeated_trials(repeat_n, model_name, data, labels, idx_train, idx_val, idx_test, device, args, logger):
    """运行多次实验，计算平均值和标准差。"""
    accs = []
    best_preds = None
    best_val_acc = 0
    for i in range(repeat_n):
        # 核心：每次重复都调用单次实验函数，确保独立性
        start_time = time.time()  # 开始计时
        acc, preds = run_single_trial(model_name, data, labels, idx_train, idx_val, idx_test, device, args, logger)
        end_time = time.time()  # 结束计时
        print(f"函数执行耗时: {end_time - start_time:.6f} 秒")
        accs.append(acc)
        val_acc = calculate_accuracy(idx_val, preds.argmax(1), labels)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_preds = preds

    mean, std = compute_mean_std(accs)
    return mean, std, best_preds

from scipy.stats import gaussian_kde

def plot_prediction_distribution_plt(predictions_dict, labels, target_label=0,dataset='Polblogs'):
    # --- 1. 数据准备 ---
    x_grid = np.linspace(0, 1, 500)

    # 假设传入的 labels 是 numpy array
    labels_np = labels
    indices_target = np.where(labels_np == target_label)[0]
    print('label',target_label,len(indices_target))

    if len(indices_target) == 0:
        print(f"错误: 数据中未找到任何标签为 '{target_label}' 的节点。")
        return

    # --- 2. 开始绘图 ---
    # 使用更大的 figsize 以容纳更大的字体
    fig, ax = plt.subplots(figsize=(8, 6))

    scores_dict = {}
    # 创建一个字典来存储用于生成CSV的数据，预先放入x_grid作为第一列
    density_data_for_csv = {'Probability (x-axis)': x_grid}
    # --- 2. 绘图与分数收集 ---
    for condition_name, preds_tensor in predictions_dict.items():
        preds_prob = torch.exp(preds_tensor)
        scores = preds_prob.detach().cpu().numpy()[indices_target, target_label]
        scores_dict[condition_name] = scores

        try:
            kde = gaussian_kde(scores)
            density = kde(x_grid)
            # 将当前计算出的密度数据存入字典，键为 condition_name
            density_data_for_csv[condition_name] = density
            print(f"  - 已收集 '{condition_name}' 的密度数据。")

            ax.plot(x_grid, density, label=condition_name, linewidth=3)
            ax.fill_between(x_grid, density, alpha=0.15)
        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"警告: 无法为 '{condition_name}' 计算KDE。错误: {e}. 跳过...")
            continue

    # --- 3. 创建DataFrame并保存为CSV文件 ---
    print("\n正在生成CSV文件...")
    try:
        # 使用收集到的数据创建 pandas DataFrame
        # DataFrame的列名将自动成为字典的键
        df = pd.DataFrame(density_data_for_csv)

        # (可选) 确保列的顺序是你想要的，例如 'Ori' 在前
        desired_order = ['Probability (x-axis)'] + list(predictions_dict.keys())
        # 过滤掉可能因计算失败而不存在的列
        existing_columns = [col for col in desired_order if col in df.columns]
        df = df[existing_columns]

        # 定义文件名并保存
        csv_filename = f"density_distributions_{dataset}_label-{target_label}.csv"
        df.to_csv(csv_filename, index=False)  # index=False 避免在文件中写入DataFrame的行索引

        print(f"密度数据已成功保存为: {csv_filename}")

    except Exception as e:
        print(f"错误: 无法创建或保存CSV文件。错误: {e}")

    # run_chi_squared_analysis(scores_dict)
    analysis_text = ""
    if 'Ori' in scores_dict and 'After Attack' in scores_dict and 'After RNE' in scores_dict:
        scores_ori = scores_dict['Ori']
        scores_att = scores_dict['After Attack']
        scores_rne = scores_dict['After RNE']

        # --- 1. 分布距离分析 (Jenson-Shannon Divergence) ---
        print("\n--- Distribution Distance Analysis (JSD) ---")
        # 为了计算JSD，我们需要先将得分数据转换成概率分布（直方图）
        # 确保所有分布使用相同的bins
        bins = np.linspace(0, 1, 51)  # 50个bins
        hist_ori, _ = np.histogram(scores_ori, bins=bins, density=True)
        hist_att, _ = np.histogram(scores_att, bins=bins, density=True)
        hist_rne, _ = np.histogram(scores_rne, bins=bins, density=True)

        # 添加一个极小值防止log(0)错误
        epsilon = 1e-10
        hist_ori += epsilon
        hist_att += epsilon
        hist_rne += epsilon

        # 计算JSD
        jsd_att_vs_ori = jensenshannon(hist_ori, hist_att)
        jsd_rne_vs_ori = jensenshannon(hist_ori, hist_rne)
        recovery_rate = (jsd_att_vs_ori - jsd_rne_vs_ori) / jsd_att_vs_ori if jsd_att_vs_ori > 0 else float('inf')

        print(f"  JSD (Ori vs Attack): {jsd_att_vs_ori:.4f}")
        print(f"  JSD (Ori vs RNE):   {jsd_rne_vs_ori:.4f}")
        print(f"  Distribution Recovery Rate: {recovery_rate:.2%}")

        analysis_text += f"JSD (vs. Ori):\n"
        analysis_text += f"  Attack: {jsd_att_vs_ori:.3f}\n"
        analysis_text += f"  Immune: {jsd_rne_vs_ori:.3f}\n\n"

        # --- 2. 提升的显著性检验 (Wilcoxon Signed-Rank Test) ---
        # 我们要检验的是 "免疫后的误差" 是否显著小于 "攻击带来的偏差"
        print("\n--- Significance of Improvement (Wilcoxon Test) ---")
        deviations_attack = np.abs(scores_ori - scores_att)
        deviations_rne = np.abs(scores_ori - scores_rne)

        # 执行Wilcoxon符号秩检验 (配对样本的非参数检验)
        # 备择假设H1: deviations_rne < deviations_attack
        # 这等价于检验: deviations_attack - deviations_rne > 0
        try:
            w_stat, p_val = wilcoxon(deviations_attack, deviations_rne, alternative='greater')
            print(f"  Wilcoxon test for improvement (one-sided):")
            print(f"  p-value = {p_val:.3g}")
            print(f"  W-statistic = {w_stat:.3g}")
            if p_val < 0.05:
                print("  Conclusion: The improvement by RNE is statistically significant.")
            else:
                print("  Conclusion: The improvement by RNE is not statistically significant.")

            analysis_text += f"Improvement Significance:\n"
            analysis_text += f"  p-value (one-sided) = {p_val:.2e}"

        except ValueError as e:
            print(f"  Could not perform Wilcoxon test: {e}")





    # # --- 3. 执行T检验并显示结果 ---
    # if 'Ori' in scores_dict:
    #     scores_ori = scores_dict['Ori']
    #     analysis_text = "T-test P-values (vs. Ori):\n\n"
    #
    #     print("\n--- T-test Results (Comparing Means) ---")
    #
    #     # 比较 'After Attack' vs 'Ori'
    #     if 'After Attack' in scores_dict:
    #         scores_att = scores_dict['After Attack']
    #         # 使用 ttest_ind 进行独立样本T检验。
    #         # equal_var=False 执行Welch's T-test，它不要求两组方差相等，更稳健。
    #         t_stat_att, p_val_att = ttest_ind(scores_ori, scores_att, equal_var=False)
    #         analysis_text += f"Attack: p = {p_val_att:.3g}, t = {t_stat_att:.3g}\n"
    #         print(f"  Ori vs. After Attack: p-value = {p_val_att:.3g}, t = {t_stat_att:.3g}")
    #
    #     # 比较 'After RNE' vs 'Ori'
    #     if 'After RNE' in scores_dict:
    #         scores_rne = scores_dict['After RNE']
    #         t_stat_rne, p_val_rne = ttest_ind(scores_ori, scores_rne, equal_var=False)
    #         analysis_text += f"\nRNE:    p = {p_val_rne:.3g}, t = {t_stat_rne:.3g}"
    #         print(f"  Ori vs. After RNE: p-value = {p_val_rne:.3g}, t = {t_stat_rne:.3g}")
    #
    #     print("--------------------------------------\n")
    #
    #     # props = dict(boxstyle='round', facecolor='wheat', alpha=0.6)
    #     # ax.text(0.97, 0.97, analysis_text, transform=ax.transAxes, fontsize=14,
    #     #         verticalalignment='top', horizontalalignment='right', bbox=props)
    #
    # --- 4. 格式化图表 ---
    ax.set_title(f"Prediction Distribution on {dataset} (Class: {target_label})", fontsize=22, weight='bold')
    ax.set_xlabel(f"Predicted Probability for Class {target_label}", fontsize=18)
    ax.set_ylabel("Density", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(title='Condition', fontsize=14, title_fontsize=16, loc='upper left')
    ax.set_xlim(0, 1);
    ax.set_ylim(bottom=0)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    # --- 5. 保存与显示 ---
    filename = f"prediction_analysis_{dataset}_label-{target_label}_with_ttest.pdf"
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    print(f"图表已成功保存为: {filename}")
    plt.show()
def run_experiment(args, device):
    """执行完整的实验流程"""
    # --- 1. 数据加载和预处理 ---
    logger = get_logger('models.log', level=0)
    pyg_data = get_data(args.dataset).to(device)
    homophily_ratio = calculate_graph_homophily(pyg_data)
    print('clean:',homophily_ratio)
    mod_adj = load_attack_adj(args.attack_name, args.dataset, args.rate)

    mod_pyg_data = pyg_data.clone()
    if args.use_attack==True:
        mod_pyg_data.adj_t = retype_adj(mod_adj)
        coo_adj = mod_adj.tocoo()
        mod_pyg_data.edge_index = torch.tensor(np.vstack((coo_adj.row, coo_adj.col)), dtype=torch.long, device=device)
    homophily_ratio = calculate_graph_homophily(mod_pyg_data)
    print('attack:', homophily_ratio)
    labels = pyg_data.y.cpu().numpy()
    idx_train = torch.nonzero(pyg_data.train_mask).squeeze()
    idx_test = torch.nonzero(pyg_data.test_mask).squeeze()
    idx_val = torch.nonzero(pyg_data.val_mask).squeeze()

    result = []

    # --- 2. 基线评估 (模型在干净图和攻击图上的初始表现) ---
    print("--- Running Baseline Evaluation ---")
    # 在干净图上的表现
    clean_mean, clean_std, preds_ori = run_repeated_trials(args.repeat_n, args.model_name, pyg_data, labels, idx_train, idx_val,
                                                   idx_test, device, args, logger)
    result.append(['clean', clean_mean, clean_std])
    print(f"Clean Graph Accuracy: {clean_mean:.4f} ± {clean_std:.4f}")

    # 在被攻击图上的表现 (Evasion Attack)
    eva_mean, eva_std, preds_att = run_repeated_trials(args.repeat_n, args.model_name, mod_pyg_data, labels, idx_train, idx_val,
                                               idx_test, device, args, logger)
    result.append(['ori_attack', eva_mean, eva_std])
    print(f"Ori Attack Accuracy on Attacked Graph: {eva_mean:.4f} ± {eva_std:.4f}")

    # --- 3. 迭代式防御流程 ---
    print("\n--- Running Iterative Defense ---")
    if args.add_node_n > 0:
        # 注入初始节点
        start_time = time.time()  # 结束计时
        iter_pyg_data = Initial_inject_nodes(mod_pyg_data, n=args.add_node_n, k=args.mean_node_n).to(device)
        end_time = time.time()  # 结束计时
        print(f"inject函数执行耗时: {end_time - start_time:.6f} 秒")

        injected_node_ids = torch.arange(pyg_data.num_nodes, iter_pyg_data.num_nodes, device=device)
        all_node_idx = torch.arange(pyg_data.num_nodes, device=device)
        # 初始时，除了训练集节点，其他所有原图节点都可被连接
        not_connected_node_idx = all_node_idx[~torch.isin(all_node_idx, idx_train)]

        # a. 防御第一步：仅注入节点后的效果
        mean, std, preds_best = run_repeated_trials(1, args.model_name, iter_pyg_data, labels, idx_train,
                                                    idx_val, idx_test, device, args, logger)
        connected_count = pyg_data.num_nodes - len(not_connected_node_idx)
        result.append([connected_count, mean, std])
        print(f"After injecting {args.add_node_n} nodes: Accuracy = {mean:.4f} ± {std:.4f}")

        # b. 防御第二步：迭代连接节点
        while len(not_connected_node_idx) > 0:
            # 使用最新的预测结果来指导连接
            start_time = time.time()  # 结束计时
            iter_pyg_data, not_connected_node_idx = connect_topk_confident_nodes_once(
                iter_pyg_data, preds_best, injected_node_ids, not_connected_node_idx, top_k=args.connect_n
            )
            end_time = time.time()  # 结束计时
            print(f"connect函数执行耗时: {end_time - start_time:.6f} 秒")
            connected_count = pyg_data.num_nodes - len(not_connected_node_idx)
            if len(not_connected_node_idx)>args.connect_n:
                # 在更新后的图上重新进行多次实验
                mean, std, preds_best = run_repeated_trials(args.repeat_n, args.model_name, iter_pyg_data, labels,
                                                            idx_train, idx_val, idx_test, device, args, logger)
            else:
                mean, std, preds_best = run_repeated_trials(args.repeat_n, args.model_name, iter_pyg_data, labels,
                                                            idx_train, idx_val, idx_test, device, args, logger)
            result.append([connected_count, mean, std])
            print(f"After connecting {connected_count} nodes: Accuracy = {mean:.4f} ± {std:.4f}")

    mean, std, preds_rne = run_repeated_trials(args.repeat_n, args.model_name, iter_pyg_data, labels,
                                                idx_train, idx_val, idx_test, device, args, logger)
    homophily_ratio = calculate_graph_homophily(iter_pyg_data)

    print('iter_pyg_data:', homophily_ratio)
    # 将预测打包成字典
    predictions = {
        'Ori': preds_ori,
        'After Attack': preds_att,
        'After RNE': preds_rne
    }



    # 1. 提取连接信息
    final_connections = {}
    injected_node_ids = torch.arange(pyg_data.num_nodes, iter_pyg_data.num_nodes)
    injected_set = set(injected_node_ids.cpu().numpy())
    edge_index = iter_pyg_data.edge_index.cpu().numpy()

    # 从最终图的 .y 张量中直接获取抗体节点的标签
    final_y_tensor = iter_pyg_data.y.cpu()

    for i in range(edge_index.shape[1]):
        u, v = edge_index[:, i]
        u_is_antibody = u in injected_set
        v_is_antibody = v in injected_set

        if u_is_antibody and not v_is_antibody:
            regular_node, antibody_node = v, u
            # 直接从最终图的标签张量中读取抗体节点的类别！
            connection_class = final_y_tensor[antibody_node].item()
            final_connections[regular_node] = connection_class
        elif v_is_antibody and not u_is_antibody:
            regular_node, antibody_node = u, v
            # 直接从最终图的标签张量中读取抗体节点的类别！
            connection_class = final_y_tensor[antibody_node].item()
            final_connections[regular_node] = connection_class

    # 2. 对所有被连接的节点进行统计分析 (这部分逻辑完全不变)
    conn_corr_pred_corr, conn_corr_pred_wrong = 0, 0
    conn_wrong_pred_corr, conn_wrong_pred_wrong = 0, 0

    for node_id, connected_class in final_connections.items():
        if node_id in idx_train.cpu().numpy():
            continue

        true_label = labels[node_id]
        # 1. 从 preds_rne 中获取单个节点的完整预测向量 (logits or probabilities)
        node_prediction_vector = preds_rne[node_id]
        # 2. 使用 argmax 找到最大值的索引，这才是最终的预测类别
        # 假设 preds_rne 是 PyTorch Tensor
        final_pred_label = torch.argmax(node_prediction_vector)

        # is_connection_correct = (connected_class == true_label)
        # is_prediction_correct = (final_pred_label == true_label)

        is_connection_correct = (
                    connected_class == true_label)  # This might already be a bool, but .item() is safer if not.
        if hasattr(is_connection_correct, 'item'):
            is_connection_correct = is_connection_correct.item()

        is_prediction_correct = (final_pred_label == true_label).item()

        if is_connection_correct and is_prediction_correct:
            conn_corr_pred_corr += 1
        elif is_connection_correct and not is_prediction_correct:
            conn_corr_pred_wrong += 1
        elif not is_connection_correct and is_prediction_correct:
            conn_wrong_pred_corr += 1
        elif not is_connection_correct and not is_prediction_correct:
            conn_wrong_pred_wrong += 1

    # 3. 打印最终的全局统计结果 (这部分逻辑完全不变)
    print("\n--- Global Connection Analysis (Final Graph) ---")
    total_connected_nodes = len(final_connections) - len(idx_train)  # 减去训练集节点数
    print(f"  Total non-training nodes connected to antibodies: {total_connected_nodes}")
    print(f"  - Nodes connected to CORRECT class: {conn_corr_pred_corr + conn_corr_pred_wrong}")
    print(f"      - And final prediction was CORRECT: {conn_corr_pred_corr}")
    print(f"      - And final prediction was WRONG:   {conn_corr_pred_wrong}")
    print(f"  - Nodes connected to WRONG class: {conn_wrong_pred_corr + conn_wrong_pred_wrong}")
    print(f"      - But final prediction was CORRECT: {conn_wrong_pred_corr}")
    print(f"      - And final prediction was WRONG:   {conn_wrong_pred_wrong}")
    print("--------------------------------------------------\n")

    # --- 4. 保存结果 ---
    # 定义结果文件夹的名称
    output_dir = "result_poisoning"
    # 创建文件夹，如果文件夹已存在则不执行任何操作，也不会报错
    os.makedirs(output_dir, exist_ok=True)
    file_name = (
        f"result_poisoning/{args.dataset}_{args.attack_name}_{args.model_name}_"
        f"attack_rate{args.rate}_add{args.add_node_n}_mean{args.mean_node_n}_conn{args.connect_n}.csv"
    )
    # 修改列名以反映实际内容
    df = pd.DataFrame(result, columns=["#Connected Nodes or Stage", "Mean Accuracy", "Std"])
    df["Mean Accuracy"] = (df["Mean Accuracy"] * 100).round(2)
    df["Std"] = (df["Std"] * 100).round(2)
    df.to_csv(file_name, index=False)
    print(f"\nResults saved to {file_name}")
    print(df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph Neural Network with Attack and Defense')
    # 参数定义保持不变
    parser.add_argument('--dataset', type=str, default='acm', help='Dataset name')
    parser.add_argument('--model_name', type=str, default='noisy',
                        choices=['gcn', 'gnn-guard', 'rgcn', 'gcn-jaccard', 'noisy', 'gcorn'], help='Model name')
    parser.add_argument('--source', type=str, default='gcn', help='Source model for attack')
    parser.add_argument('--attack_name', type=str, default='Metattack',
                        choices=['greedy', 'prbcd', 'pga', 'pgdattack-CW', 'Metattack', 'MinMax', 'DICE'],
                        help='Attack method')
    parser.add_argument('--rate', type=float, default=0.25, help='Attack perturbation rate')
    parser.add_argument('--add_node_n', type=int, default=5, help='Number of nodes to add')
    parser.add_argument('--mean_node_n', type=int, default=5, help='Mean number of nodes for initialization')
    parser.add_argument('--connect_n', type=int, default=500, help='Number of connections for top-k confident nodes')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--repeat_n', type=int, default=1, help='Number of repetitions for each experiment')
    parser.add_argument('--use_attack', default=True,
                        help='Use attack (poisoning) if True, else evasion')

    args = parser.parse_args()
    device = get_device(args.gpu_id)

    # 运行主实验流程
    run_experiment(args, device)