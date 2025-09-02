# This software is for non-commercial use only.
# Commercial use requires a separate license.

import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import logging
import threading
import queue
import re
from utils import load_problems

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

task_queue = queue.Queue()


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
        problem_num += args.problem_start_idx
        logging.info(f"[BigModel] Processing problem {problem_num}, alpha={alpha}")

        problem_output_dir = os.path.join(generations_base_dir, f"problem_{problem_num}")
        os.makedirs(problem_output_dir, exist_ok=True)

        prompt = problem + "\nPlease reason step by step, and put your final answer within \\boxed{}.\n<think>\n"
        output, new_token_count = generate_with_intervention(
            model, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens,
            intervention_vector=intervention_vector,
            intervention_scale=alpha
        )

        # 如果达到 max_new_tokens，就认为被截断，并且末尾没有自然结束标志
        if new_token_count >= args.max_new_tokens and not output.endswith(("<eos>", ".", "!", "?")):
            output = output.rstrip() + " <eos>"
            logging.warning(f"[BigModel] Output for problem {problem_num}, alpha={alpha} "
                            f"was truncated at {args.max_new_tokens} tokens.")

        output_file_path = os.path.join(problem_output_dir, f"response_alpha_{alpha}.txt")
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