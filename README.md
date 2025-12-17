# Adaptive Overclocking: Dynamic Control of Thinking Path Length via Real-Time Reasoning Signals

![Teaser](./readme_resources/new_teaser.jpg)
*Adaptive Overclocking: Dynamically adjusting reasoning speed by leveraging real-time uncertainty signals and input complexity estimation to accelerate thinking on simple problems while maintaining caution on complex ones.*

## 📄 Paper Information[https://arxiv.org/pdf/2509.17000]

> **Title**: Adaptive Overclocking: Dynamic Control of Thinking Path Length via Real-Time Reasoning Signals  
> **Authors**: Shuhao Jiang, Songbo Wang, Yang Qiao, Chun Xu, Chaoyang Zheng, Shengyi Zhou, Huanjun Wang, Fangming Li, Cong Zhang, Jiyu Wang (Huawei Technologies Co., Ltd.)  
> **Abstract**: Large Reasoning Models (LRMs) often suffer from computational inefficiency due to "overthinking." To address this, we propose **Adaptive Overclocking**, a method that makes the overclocking hyperparameter dynamic and context-aware. Our method adjusts reasoning speed in real-time through two complementary signals: (1) **Complexity-Guided Alpha Initialization (CG-αI)** based on input difficulty, and (2) **Uncertainty-Aware Alpha Scheduling (UA-αS)** based on token-level model uncertainty. We combine these into a **Hybrid Adaptive Control (HAC)** system. Experiments on GSM8K, MATH, and SVAMP show that HAC achieves superior accuracy-latency trade-offs.

---

## 🚀 Core Method: Hybrid Adaptive Control (HAC)

This repository implements the HAC framework proposed in the paper, which consists of the following components:

1.  **Thinking Progress Vector (TPV)**: The backbone component that guides the model to accelerate thinking in the hidden state space (based on Eisenstadt et al., 2025).
2.  **Complexity-Guided Alpha Initialization (CG-αI)**: Uses a lightweight router model (Small Model) to estimate problem difficulty (Easy, Medium, Hard) and assigns a global initial acceleration strength $\alpha_{init}$ accordingly.
3.  **Uncertainty-Aware Alpha Scheduling (UA-αS)**: Dynamically adjusts $\alpha_t$ based on token-level predictive uncertainty (entropy) during generation. It slows down when the model is uncertain and accelerates when it is confident.

---

## 🛠️ Installation

```bash
conda create -n adaptive-overclocking python=3.10
conda activate adaptive-overclocking
conda install pytorch==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt

## 🔄 Workflow Overview
The workflow consists of two main phases:

- TPV Training Phase: Generating data and training the base TPV linear regression model.

- Adaptive Overclocking Phase: Using the trained TPV vector combined with HAC strategies for dynamic inference.

### Phase 1: TPV Monitoring Pipeline (Training)
**Step 1: Generate TPV Data**

Generate model responses and extract hidden states.

```bash
python generate_tpv_data.py \
  --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
  --max_new_tokens 1024 \
  --dataset "math500" \
  --start_problem_index 0 \
  --end_problem_index 80 \
  --output_dir "hidden_states_output"

**Step 2: Prepare Dataset**
Process hidden states to build the regression dataset.
```bash
python prepare_tpv_dataset.py \
  --input_dir "hidden_states_output" \
  --output_dir "qwen_math_tpv_dataset" \
  --end_of_thinking_token "</think>"

**Step 3: Train TPV Regressor**
Train the linear regression model to obtain the TPV vector (``tpv_linear_weights.npy``).
```bash
python train_tpv.py \
  --input_dir "qwen_math_tpv_dataset" \
  --output_dir "qwen_math_tpv_model" \
  --device "cuda"

### Phase 2: HAC Intervention System (Inference)
This is the core implementation of our paper. Use the hac.py script to run Hybrid Adaptive Control (HAC). This script launches a lightweight model (for difficulty assessment) and a large reasoning model (for solving problems) in parallel.
```bash
python hac.py \
  --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
  --intervention_vector_path "qwen_math_tpv_model/tpv_linear_weights.npy" \
  --dataset "math500" \
  --output_generations_dir "hac_results" \
  --task_name "adaptive_run" \
  --enable_uaas \
  --range 40.0 \
  --uaas_k 20.0 \
  --uaas_threshold 0.015

**Arguments Explanation:**
- ``--enable_uaas``: Enables Uncertainty-Aware Alpha Scheduling. If omitted, only static or complexity-based initialization is used.
- ``--range``: The range of dynamic adjustment for UA-αS (i.e., $\alpha_{max} = \alpha_{base} + \text{range}$).
- ``--uaas_k``: The steepness parameter for the sigmoid function, controlling sensitivity to uncertainty.
- ``--uaas_threshold``: The uncertainty threshold.
- Note: The script automatically loads a small model (e.g., Qwen/Qwen3-4B-Instruct) internally as the Complexity Router for CG-αI.

## Run Static Intervention (Baseline)
If you wish to run the standard static Overclocking for comparison:
```bash
python tpv_intervention.py \
  --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
  --intervention_vector_path "qwen_math_tpv_model/tpv_linear_weights.npy" \
  --alpha 100.0

## 🔗 Citation
```bibtex
@misc{eisenstadt2025overclockingllmreasoningmonitoring,
      title={Overclocking LLM Reasoning: Monitoring and Controlling Thinking Path Lengths in LLMs}, 
      author={Roy Eisenstadt and Itamar Zimerman and Lior Wolf},
      year={2025},
      eprint={2506.07240},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.07240}, 
}
We also acknowledge the foundational work on Thinking Progress Vectors:
```bibtex
@misc{eisenstadt2025overclockingllmreasoningmonitoring,
      title={Overclocking LLM Reasoning: Monitoring and Controlling Thinking Path Lengths in LLMs}, 
      author={Roy Eisenstadt and Itamar Zimerman and Lior Wolf},
      year={2025},
      eprint={2506.07240},
}
## License
This repository is licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.