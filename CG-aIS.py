# This software is for non-commercial use only.
# Commercial use requires a separate license.

import os
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import logging
import threading
import queue
import re
import math
from utils import load_problems

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

task_queue = queue.Queue()


def compute_token_entropy(logits, vocab_size):
    """
    计算token的归一化熵
    Args:
        logits: 模型输出的logits [batch_size, vocab_size]
        vocab_size: 词汇表大小
    Returns:
        normalized_entropy: 归一化熵 u_t = Entropy(P(token_t)) / log(|V|)
    """
    # 确保logits是float32类型以提高数值稳定性
    if logits.dtype == torch.float16:
        logits = logits.float()
    
    # 数值稳定的softmax计算
    # 减去最大值以避免overflow
    logits_shifted = logits - torch.max(logits, dim=-1, keepdim=True)[0]
    
    # 计算概率分布
    probs = F.softmax(logits_shifted, dim=-1)
    
    # 检查概率分布的有效性
    if torch.any(torch.isnan(probs)) or torch.any(torch.isinf(probs)):
        # 如果softmax结果无效，返回中等不确定性
        return torch.tensor(0.5, device=logits.device, dtype=torch.float32)
    
    # 计算熵，使用更大的epsilon避免log(0)
    log_probs = torch.log(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    
    # 检查熵的有效性
    if torch.any(torch.isnan(entropy)) or torch.any(torch.isinf(entropy)):
        return torch.tensor(0.5, device=logits.device, dtype=torch.float32)
    
    # 归一化 - 理论最大熵是log(vocab_size)
    max_entropy = math.log(vocab_size)
    normalized_entropy = entropy / max_entropy
    
    # 确保结果在[0,1]范围内
    normalized_entropy = torch.clamp(normalized_entropy, 0.0, 1.0)
    
    return normalized_entropy


def compute_dynamic_alpha(uncertainty, alpha_base, alpha_max=120.0, k=10.0, u_threshold=0.5):
    """
    使用sigmoid函数将不确定性映射到alpha值
    Args:
        uncertainty: 归一化的token熵
        alpha_base: 基础alpha值（由小模型给出）
        alpha_max: 最大alpha值
        k: sigmoid函数的陡峭度参数
        u_threshold: 不确定性阈值
    Returns:
        alpha_t: 动态调整的alpha值
    """
    sigmoid_term = torch.sigmoid(k * (uncertainty - u_threshold))
    alpha_t = alpha_base + (alpha_max - alpha_base) * (1 - sigmoid_term)
    return alpha_t


def generate_with_intervention(
    model,
    tokenizer,
    prompt,
    max_new_tokens=1,
    intervention_vector=None,
    intervention_scale=1.0
):
    """Generates tokens with intervention by hooking into the generation process."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_token_count = inputs.input_ids.size(-1)

    hook_handles = []

    def forward_hook(module, input_args, output):
        original_output = output
        if isinstance(output, tuple):
            hidden_states_to_modify = output[0]
        elif isinstance(output, torch.Tensor):
            hidden_states_to_modify = output
        else:
            logging.warning(f"Unexpected output type from hooked layer: {type(output)}")
            return output

        modified_states = hidden_states_to_modify.clone()
        reshaped_vector = intervention_vector.view(1, 1, -1).to(modified_states.device)
        modified_states[:, -1:, :] = modified_states[:, -1:, :] + reshaped_vector * intervention_scale

        if isinstance(original_output, tuple):
            return (modified_states,) + original_output[1:]
        else:
            return modified_states

    # 注册 hook
    target_intervention_module = model.model.norm
    if intervention_vector is not None:
        hook_handles.append(target_intervention_module.register_forward_hook(forward_hook))

    try:
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        # 解码
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 统计 token 使用
        total_token_count = outputs.size(-1)
        new_token_count = max(0, total_token_count - input_token_count)
        logging.info(
            f"Token usage - prompt_tokens: {input_token_count}, "
            f"generated_tokens: {new_token_count}, total_tokens: {total_token_count}"
        )
    finally:
        for h in hook_handles:
            h.remove()

    # 返回 (生成文本, 新生成 token 数)
    return result, new_token_count


def generate_with_uaas_intervention(
    model,
    tokenizer,
    prompt,
    max_new_tokens=1,
    intervention_vector=None,
    alpha_base=50.0,
    alpha_max=100.0,
    k=10.0,
    u_threshold=0.5,
    enable_uaas=True
):
    """
    使用UA-aS方法高效生成tokens，手动管理KV缓存，避免重复计算。
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    input_token_count = input_ids.size(1)
    vocab_size = model.config.vocab_size

    generated_tokens = []
    alpha_values = []
    uncertainty_values = []
    
    past_key_values = None
    
    # 确保模型处于评估模式
    model.eval()

    # 初始前向传播，获取第一个logits和KV缓存
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True, return_dict=True)
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

    for step in range(max_new_tokens):
        hook_handles = []
        current_alpha = alpha_base
        
        # 1. 根据上一步的logits计算当前token的alpha
        if enable_uaas and intervention_vector is not None:
            # 确保logits是float32以提高数值稳定性
            stable_logits = logits.float() if logits.dtype == torch.float16 else logits
            
            uncertainty = compute_token_entropy(stable_logits, vocab_size)
            uncertainty_val = uncertainty.item()

            if math.isnan(uncertainty_val) or math.isinf(uncertainty_val):
                logging.warning(f"Invalid uncertainty value: {uncertainty_val}, using default alpha_base")
                current_alpha = alpha_base
                uncertainty_val = 0.5
            else:
                sigmoid_val = 1 / (1 + math.exp(-k * (uncertainty_val - u_threshold)))
                current_alpha = alpha_base + (alpha_max - alpha_base) * (1 - sigmoid_val)

            uncertainty_values.append(uncertainty_val)
            alpha_values.append(current_alpha)
            
            if step < 5:
                logging.info(f"Token {step}: uncertainty={uncertainty_val:.4f}, alpha={current_alpha:.2f}")

        # 2. 从上一步的logits中选择下一个token
        next_token_id = torch.argmax(logits, dim=-1).unsqueeze(-1)

        # 记录生成的token
        generated_tokens.append(next_token_id.item())
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)

        # 检查是否生成了结束token
        if next_token_id.item() == tokenizer.eos_token_id:
            break

        # 3. 为下一步的前向传播设置hook
        def make_hook(alpha_val):
            def forward_hook(module, input_args, output):
                hidden_states_to_modify = output[0] if isinstance(output, tuple) else output
                reshaped_vector = intervention_vector.view(1, 1, -1).to(hidden_states_to_modify.device)
                modified_states = hidden_states_to_modify.clone()
                modified_states[:, -1:, :] += reshaped_vector * alpha_val
                return (modified_states,) + output[1:] if isinstance(output, tuple) else modified_states
            return forward_hook
        
        if intervention_vector is not None:
            target_intervention_module = model.model.norm
            forward_hook = make_hook(current_alpha)
            hook_handles.append(target_intervention_module.register_forward_hook(forward_hook))

        # 4. 执行单步前向传播，传入新的token和过去的KV缓存
        try:
            with torch.no_grad():
                # 注意：这里只传入最后一个token，并附带past_key_values
                outputs = model(
                    input_ids=next_token_id, 
                    past_key_values=past_key_values, 
                    use_cache=True, 
                    return_dict=True
                )
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
        finally:
            for h in hook_handles:
                h.remove()

    # 解码和日志记录部分与你原来的函数保持一致
    result = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    new_token_count = len(generated_tokens)
    
    # (此处省略与你原代码相同的日志记录部分，可以直接复制过来)
    if enable_uaas and alpha_values and len(alpha_values) > 0:
        valid_alphas = [a for a in alpha_values if not (math.isnan(a) or math.isinf(a))]
        valid_uncertainties = [u for u in uncertainty_values if not (math.isnan(u) or math.isinf(u))]
        
        if valid_alphas and valid_uncertainties:
            avg_alpha = sum(valid_alphas) / len(valid_alphas)
            avg_uncertainty = sum(valid_uncertainties) / len(valid_uncertainties)
            logging.info(
                f"UA-aS Stats - avg_alpha: {avg_alpha:.2f}, avg_uncertainty: {avg_uncertainty:.4f}, "
                f"alpha_range: [{min(valid_alphas):.2f}, {max(valid_alphas):.2f}]"
            )
    
    logging.info(
        f"Token usage - prompt_tokens: {input_token_count}, "
        f"generated_tokens: {new_token_count}, total_tokens: {input_ids.size(-1)}"
    )

    return result, new_token_count


difficulty_to_alpha = {
    0.1: 100,  # Very Easy
    0.3: 80,   # Easy
    0.5: 60,   # Medium
    0.7: 40,   # Relatively Hard
    1.0: 20    # Hard
}

def evaluate_problem(problem, small_model, small_tokenizer):
    """Assign alpha using small model (0.1, 0.3, 0.5, 0.7, 1.0)."""
    prompt = f"""
You are an expert mathematics evaluator. 
Your task is to assign a difficulty score to the given problem, 
based on the following 5 categories:

- Very Easy (typical high-school exercise) → 0.1
- Easy (standard competition warm-up or basic calculus/algebra) → 0.3
- Medium (Olympiad-style but solvable with standard techniques) → 0.5
- Relatively Hard (hard Olympiad problems or advanced university-level) → 0.7
- Hard (research-level or extremely challenging Olympiad problems) → 1.0

Important rules:
- Output ONLY the difficulty score as a number.
- Do NOT include any explanation or extra text.
- Write only the number on one line.

Here are some examples for reference (with analysis for why they are rated this way):

Problem: For what value of $x$ is $2^3\\cdot3^x=72$?
Analysis: Factorization is straightforward and requires only basic algebra; typical high-school exercise.
Difficulty Score: 0.1
---
Problem: Solve for $x: 3^(2x) + 19 = 10^x$.
Analysis: Requires recognizing monotonicity and comparing exponential functions; standard competition-level skill.
Difficulty Score: 0.3
---
Problem: Ten treeks weigh as much as three squigs and one goolee. Two treeks and one goolee are equal in weight to one squig. The combined weight of how many treeks equals the weight of one squig?
Analysis: Involves solving a small system of linear equations with abstract variables; medium difficulty.
Difficulty Score: 0.5
---
Problem: A right cylindrical tank with circular bases is being filled with water at $20\\pi$ m³/h. As the tank fills, water rises 4 m/h. Find the radius of the tank.
Analysis: Involves relating volume rate to height rate; requires setting up equation and solving quadratic; moderately challenging.
Difficulty Score: 0.7
---
Problem: Point $A$ lies within or on the square with corners at $(0,0)$ and $(2,2)$. Point $B$ lies within or on the square with corners at $(4,2)$ and $(5,3)$. What is the greatest possible value of the slope of the line containing points $A$ and $B$?
Analysis: Requires maximizing slope by geometric reasoning; nontrivial and very challenging; hard Olympiad-level problem.
Difficulty Score: 1.0
---

Important:Output ONLY the number (0.1,0.3,0.5,0.7,1.0) in last line. No extra words, letters, or punctuation.

Now, assess the following problem:

Problem: "{problem}"
Answer:
"""
    messages = [{"role": "user", "content": prompt}]
    text = small_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    inputs = small_tokenizer([text], return_tensors="pt").to(small_model.device)
    outputs = small_model.generate(**inputs, max_new_tokens=32)
    output_ids = outputs[0][len(inputs.input_ids[0]):].tolist()
    content = small_tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    match = re.search(r"(0\.1|0\.3|0\.5|0\.7|1\.0)", content)
    difficulty = float(match.group(0)) if match else 0.5  # 默认中等
    return difficulty


# -------------------------
# 1. small_model_worker 保持不变
# -------------------------
def small_model_worker(problems, small_model, small_tokenizer):
    """Produce tasks: (problem_num, problem, alpha) using discrete mapping."""
    logging.info("[SmallModel] Worker started.")
    for i, problem in enumerate(problems):
        difficulty = evaluate_problem(problem, small_model, small_tokenizer)
        alpha_discrete = difficulty_to_alpha.get(difficulty, 60)  # 默认中等
        logging.info(f"[SmallModel] Problem {i} -> difficulty={difficulty} -> alpha={alpha_discrete}")
        task_queue.put((i, problem, alpha_discrete))
    task_queue.put(None)  # 结束信号

# -------------------------
# 2. big_model_worker 保持阻塞式消费
# -------------------------
def big_model_worker(model, tokenizer, intervention_vector, args, generations_base_dir):
    """Consume tasks and run big model with intervention."""
    while True:
        item = task_queue.get()
        if item is None:
            task_queue.put(None)  # 如果有多个big_model线程，需要再次放回None
            break
        problem_num, problem, alpha = item
        logging.info(f"[BigModel] Processing problem {problem_num}, alpha_base={alpha}")

        problem_output_dir = os.path.join(generations_base_dir, f"problem_{problem_num}")
        os.makedirs(problem_output_dir, exist_ok=True)

        prompt = problem + "\nPlease reason step by step, and put your final answer within \\boxed{}.\n<think>\n"
        
        # 根据enable_uaas参数选择生成方法
        if args.enable_uaas:
            output, new_token_count = generate_with_uaas_intervention(
                model, tokenizer, prompt,
                max_new_tokens=args.max_new_tokens,
                intervention_vector=intervention_vector,
                alpha_base=alpha,
                alpha_max=args.alpha_max,
                k=args.uaas_k,
                u_threshold=args.uaas_threshold,
                enable_uaas=True
            )
            # 文件名包含uaas标识
            output_file_path = os.path.join(problem_output_dir, f"response_uaas_alpha_base_{alpha}.txt")
        else:
            # 使用原来的固定alpha方法
            output, new_token_count = generate_with_intervention(
                model, tokenizer, prompt,
                max_new_tokens=args.max_new_tokens,
                intervention_vector=intervention_vector,
                intervention_scale=alpha
            )
            output_file_path = os.path.join(problem_output_dir, f"response_alpha_{alpha}.txt")

        # 如果达到 max_new_tokens，就认为被截断，并且末尾没有自然结束标志
        if new_token_count >= args.max_new_tokens and not output.endswith(("<eos>", ".", "!", "?")):
            output = output.rstrip() + " <eos>"
            logging.warning(f"[BigModel] Output for problem {problem_num}, alpha_base={alpha} "
                            f"was truncated at {args.max_new_tokens} tokens.")

        try:
            with open(output_file_path, "w") as f:
                f.write(output)
            logging.info(f"[BigModel] Saved {output_file_path}")
        except IOError as e:
            logging.error(f"Failed to write output file {output_file_path}: {e}")





def main():
    parser = argparse.ArgumentParser(description="Generate text with model intervention using alpha from small model.")
    parser.add_argument("--model_name_or_path", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--intervention_vector_path", type=str, default="tpv_model/tpv_linear_weights.npy")
    parser.add_argument("--dataset", type=str, default="math500")
    parser.add_argument("--output_generations_dir", type=str, default="llama_intervention_responses")
    parser.add_argument("--task_name", type=str, default="math500")
    parser.add_argument("--problem_start_idx", type=int, default=0)
    parser.add_argument("--problem_end_idx", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--torch_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--alpha", type=float, default=50, help="initial alpha")
    
    # UA-aS 相关参数
    parser.add_argument("--enable_uaas", action="store_true", help="Enable Uncertainty-Aware adaptive Strength (UA-aS)")
    parser.add_argument("--alpha_max", type=float, default=200.0, help="Maximum alpha value for UA-aS")
    parser.add_argument("--uaas_k", type=float, default=100.0, help="Steepness parameter k for sigmoid function")
    parser.add_argument("--uaas_threshold", type=float, default=0.02, help="Uncertainty threshold u_threshold")
    
    args = parser.parse_args()

    if args.tokenizer_name_or_path is None:
        args.tokenizer_name_or_path = args.model_name_or_path

    # Load big model
    logging.info(f"Loading big model {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_dtype = getattr(torch, args.torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map=args.device if args.device != "auto" else "auto",
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True
    )
    effective_device = model.device
    logging.info(f"Big model loaded on {effective_device}")

    # Intervention vector
    intervention_vector = torch.Tensor(np.load(args.intervention_vector_path)).to(effective_device).squeeze()
    if intervention_vector.shape[0] != model.config.hidden_size:
        raise ValueError("Intervention vector dimension mismatch")

    # Load small model
    small_model_name = "Qwen/Qwen3-4B-Instruct-2507"
    logging.info(f"Loading small model {small_model_name}")
    small_tokenizer = AutoTokenizer.from_pretrained(small_model_name)
    small_model = AutoModelForCausalLM.from_pretrained(
        small_model_name,
        torch_dtype=torch.float16,
        device_map="auto"  # 可以改成另一张卡
    )

    # Dataset
    problems = load_problems(args.dataset, args.problem_start_idx, args.problem_end_idx)

    # Output dir
    model_folder_name = args.model_name_or_path.split("/")[-1]
    generations_base_dir = os.path.join(args.output_generations_dir, model_folder_name, args.task_name, "intervention")

    # Start threads
    t_big = threading.Thread(target=big_model_worker,
                             args=(model, tokenizer, intervention_vector, args, generations_base_dir))
    t_small = threading.Thread(target=small_model_worker, args=(problems, small_model, small_tokenizer))

    # 注意：先启动 big_model_worker 线程，让它阻塞等待任务
    t_small.start()
    t_big.start()

    t_small.join()
    t_big.join()

    logging.info("Processing complete.")


if __name__ == "__main__":
    main()