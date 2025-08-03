## How to Run Experiments

The framework is built around two primary execution scripts for different attack scenarios:

* `new_poisoning_main.py`: Simulates **poisoning attacks**, where the attacker perturbs the graph *before* the model is trained.

* `new_evasion_main.py`: Simulates **evasion attacks**, where the attacker perturbs the graph *after* the model has been trained, targeting the inference phase.

### Command-Line Arguments

The behavior of the scripts is controlled via the following command-line arguments:

| Argument | Type | Default | Description | 
| --- | --- | --- | --- | 
| `--dataset` | `str` | `acm` | Name of the dataset to use. | 
| `--model_name` | `str` | `noisy` | The GNN defense model to be evaluated. Choices: `gcn`, `gnn-guard`, `rgcn`, `gcn-jaccard`, `noisy`, `gcorn`. | 
| `--source` | `str` | `gcn` | The surrogate model used to generate the attack. | 
| `--attack_name` | `str` | `Metattack` | The adversarial attack method. Choices: `greedy`, `prbcd`, `pga`, `pgdattack-CW`, `Metattack`, `MinMax`, `DICE`. | 
| `--rate` | `float` | `0.25` | Perturbation rate for the attack (e.g., percentage of edges to modify). | 
| `--add_node_n` | `int` | `5` | Number of nodes to add. | 
| `--mean_node_n` | `int` | `5` | Mean number of nodes for initialization. | 
| `--connect_n` | `int` | `500` | Number of connections for top-k confident nodes. | 
| `--gpu_id` | `int` | `0` | ID of the GPU to use for the experiment. | 
| `--repeat_n` | `int` | `1` | Number of times to repeat each experiment for robust results. | 
| `--use_attack` | `bool` | `True` | Flag to distinguish attack scenarios. `True` for poisoning, `False` for attack-free. | 

## Execution Examples

### 1. Poisoning Attack Evaluation

This example evaluates the `noisy` GNN model against a `Metattack` poisoning attack on the `acm` dataset with a `0.25` perturbation rate. The attack is generated using a `gcn` surrogate model.

```bash
python new_poisoning_main.py \
    --dataset acm \
    --model_name noisy \
    --source gcn \
    --attack_name Metattack \
    --rate 0.25 \
    --gpu_id 0 \
    --repeat_n 3
*(Note: `--use_attack` is `True` by default in `new_poisoning_main.py`)*

### 2. Evasion Attack Evaluation

This example evaluates the `gcorn` model against a `PGD` evasion attack on the `citeseer` dataset. Note that for evasion attacks, `new_evasion_main.py` should be used, which internally sets `--use_attack` to `False`.

```bash
python new_evasion_main.py \
    --dataset citeseer \
    --model_name gcorn \
    --attack_name pgdattack-CW \
    --rate 0.1 \
    --gpu_id 0