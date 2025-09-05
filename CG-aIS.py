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
    if logits.dtype == torch.float16:
        logits = logits.float()
    logits_shifted = logits - torch.max(logits, dim=-1, keepdim=True)[0]
    probs = F.softmax(logits_shifted, dim=-1)
    if torch.any(torch.isnan(probs)) or torch.any(torch.isinf(probs)):
        return torch.tensor(0.5, device=logits.device, dtype=torch.float32)
    log_probs = torch.log(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    if torch.any(torch.isnan(entropy)) or torch.any(torch.isinf(entropy)):
        return torch.tensor(0.5, device=logits.device, dtype=torch.float32)
    max_entropy = math.log(vocab_size)
    normalized_entropy = entropy / max_entropy
    normalized_entropy = torch.clamp(normalized_entropy, 0.0, 1.0)
    return normalized_entropy


def compute_dynamic_alpha(uncertainty, alpha_base, alpha_max=120.0, k=10.0, u_threshold=0.5):
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
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_token_count = inputs.input_ids.size(-1)
    hook_handles = []

    def forward_hook(module, input_args, output):
        hidden_states_to_modify = output[0] if isinstance(output, tuple) else output
        reshaped_vector = intervention_vector.view(1, 1, -1).to(hidden_states_to_modify.device)
        modified_states = hidden_states_to_modify.clone()
        modified_states[:, -1:, :] = modified_states[:, -1:, :] + reshaped_vector * intervention_scale
        return (modified_states,) + output[1:] if isinstance(output, tuple) else modified_states

    if intervention_vector is not None:
        hook_handles.append(model.model.norm.register_forward_hook(forward_hook))

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
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        total_token_count = outputs.size(-1)
        new_token_count = max(0, total_token_count - input_token_count)
    finally:
        for h in hook_handles:
            h.remove()

    return result, new_token_count


def generate_with_uaas_intervention(
    model,
    tokenizer,
    prompt,
    max_new_tokens=1,
    intervention_vector=None,
    alpha_base=50.0,
    # alpha_max=100,
    k=10.0,
    u_threshold=0.5,
    enable_uaas=True
):
    alpha_max = alpha_base + 20
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    input_token_count = input_ids.size(1)
    vocab_size = model.config.vocab_size

    generated_tokens = []
    alpha_values = []
    uncertainty_values = []
    past_key_values = None
    model.eval()

    with torch.no_grad():
        outputs = model(input_ids, use_cache=True, return_dict=True)
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

    for step in range(max_new_tokens):
        hook_handles = []
        current_alpha = alpha_base

        if enable_uaas and intervention_vector is not None:
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

        next_token_id = torch.argmax(logits, dim=-1).unsqueeze(-1)
        generated_tokens.append(next_token_id.item())
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)

        if next_token_id.item() == tokenizer.eos_token_id:
            break

        def make_hook(alpha_val):
            def forward_hook(module, input_args, output):
                hidden_states_to_modify = output[0] if isinstance(output, tuple) else output
                reshaped_vector = intervention_vector.view(1, 1, -1).to(hidden_states_to_modify.device)
                modified_states = hidden_states_to_modify.clone()
                modified_states[:, -1:, :] += reshaped_vector * alpha_val
                return (modified_states,) + output[1:] if isinstance(output, tuple) else modified_states
            return forward_hook

        if intervention_vector is not None:
            forward_hook = make_hook(current_alpha)
            hook_handles.append(model.model.norm.register_forward_hook(forward_hook))

        try:
            with torch.no_grad():
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

    result = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    new_token_count = len(generated_tokens)

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


# ------------------------- 三分类逻辑 -------------------------
def map_level_5_to_3(level_5: int) -> int:
    if level_5 in [1, 2]:
        return 1
    elif level_5 == 3:
        return 2
    elif level_5 in [4, 5]:
        return 3
    else:
        raise ValueError(f"未知 level_5: {level_5}")

three_class_alpha = {
    1: 50,  # relatively easy
    2: 30,   # medium
    3: 10    # relatively hard
}


def evaluate_problem(problem, small_model, small_tokenizer):
    """返回三分类整数 1/2/3"""
    prompt = f"""
You are an expert mathematics evaluator. 
Your task is to assign a difficulty level to the given problem (1,2,3):
1 = relatively easy (math500 1-2)
2 = medium (math500 3)
3 = relatively hard (math500 4-5)

Important rules:
- Output ONLY the difficulty level as a single number (1,2,3).
- Do NOT include explanation or extra text.

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
    outputs = small_model.generate(**inputs, max_new_tokens=16)
    output_ids = outputs[0][len(inputs.input_ids[0]):].tolist()
    content = small_tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    match = re.search(r"(1|2|3)", content)
    difficulty_class = int(match.group(0)) if match else 2
    return difficulty_class


def small_model_worker(problems, small_model, small_tokenizer):
    logging.info("[SmallModel] Worker started.")
    for i, problem in enumerate(problems):
        difficulty_class = evaluate_problem(problem, small_model, small_tokenizer)
        alpha_discrete = three_class_alpha.get(difficulty_class, 60)
        logging.info(f"[SmallModel] Problem {i} -> class={difficulty_class} -> alpha={alpha_discrete}")
        task_queue.put((i, problem, alpha_discrete))
    task_queue.put(None)


def big_model_worker(model, tokenizer, intervention_vector, args, generations_base_dir):
    while True:
        item = task_queue.get()
        if item is None:
            task_queue.put(None)
            break
        problem_num, problem, alpha = item
        problem_num += args.problem_start_idx
        logging.info(f"[BigModel] Processing problem {problem_num}, alpha_base={alpha}")

        problem_output_dir = os.path.join(generations_base_dir, f"problem_{problem_num}")
        os.makedirs(problem_output_dir, exist_ok=True)
        prompt = problem + "\nPlease reason step by step, and put your final answer within \\boxed{}.\n<think>\n"

        if args.enable_uaas:
            output, new_token_count = generate_with_uaas_intervention(
                model, tokenizer, prompt,
                max_new_tokens=args.max_new_tokens,
                intervention_vector=intervention_vector,
                alpha_base=alpha,
                # alpha_max=args.alpha_max,
                k=args.uaas_k,
                u_threshold=args.uaas_threshold,
                enable_uaas=True
            )
            output_file_path = os.path.join(problem_output_dir, f"response_uaas_alpha_base_{alpha}.txt")
        else:
            output, new_token_count = generate_with_intervention(
                model, tokenizer, prompt,
                max_new_tokens=args.max_new_tokens,
                intervention_vector=intervention_vector,
                intervention_scale=alpha
            )
            output_file_path = os.path.join(problem_output_dir, f"response_alpha_{alpha}.txt")

        if new_token_count >= args.max_new_tokens:
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

    # UA-aS parameters
    parser.add_argument("--enable_uaas", action="store_true", default=False,
                        help="Enable Uncertainty-Aware adaptive Strength (UA-aS)")
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
        device_map="auto"
    )

    # Load dataset
    problems = load_problems(args.dataset, args.problem_start_idx, args.problem_end_idx)

    # Output dir
    model_folder_name = args.model_name_or_path.split("/")[-1]
    generations_base_dir = os.path.join(args.output_generations_dir, model_folder_name, args.task_name, "intervention")

    # Start threads
    t_big = threading.Thread(target=big_model_worker,
                             args=(model, tokenizer, intervention_vector, args, generations_base_dir))
    t_small = threading.Thread(target=small_model_worker, args=(problems, small_model, small_tokenizer))

    # Start threads: small model produces tasks, big model consumes
    t_small.start()
    t_big.start()

    t_small.join()
    t_big.join()

    logging.info("Processing complete.")


if __name__ == "__main__":
    main()

