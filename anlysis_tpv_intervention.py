#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析TPV intervention结果的统计脚本
统计指标：
1. #Correct - 模型产生正确最终答案的问题数量（通过\\boxed{}中的表达式判断）
2. #Answered - 模型产生了答案的生成数量（通过\\boxed{}符号的存在判断）
3. #Ended - 生成了答案框，并且在答案框后以句号结尾的生成数量
"""

import os
import re
import json
import argparse
from pathlib import Path
import logging
import glob

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_closest_alpha_file(problem_dir: Path, target_alpha: float):
    """
    在 problem_dir 中查找与 target_alpha 最接近的 response_alpha 或 response_uaas_alpha_base_* 文件
    返回 Path 或 None
    """
    # 支持两种文件名模式
    files = list(problem_dir.glob("response_alpha_*.txt")) + list(problem_dir.glob("response_uaas_alpha_base_*.txt"))
    if not files:
        return None

    closest_file = None
    min_diff = float('inf')

    for f in files:
        # 匹配两种模式
        match = re.search(r'response_alpha_([0-9.]+)\.txt', f.name)
        if not match:
            match = re.search(r'response_uaas_alpha_base_([0-9.]+)\.txt', f.name)
        if match:
            try:
                alpha_val = float(match.group(1))
                diff = abs(alpha_val - target_alpha)
                if diff < min_diff:
                    min_diff = diff
                    closest_file = f
            except ValueError:
                continue

    return closest_file

def extract_boxed_answer(text: str):
    """
    从文本中提取最后一个 \boxed{} 中的答案。
    能够正确处理嵌套的大括号。
    返回一个元组 (answer_content, end_brace_position) 或 None。
    """
    last_match_pos = text.rfind(r'\boxed{')
    if last_match_pos == -1:
        return None

    start_pos = last_match_pos + len(r'\boxed{') - 1

    brace_count = 0
    pos = start_pos

    while pos < len(text):
        if text[pos] == '{':
            brace_count += 1
        elif text[pos] == '}':
            brace_count -= 1
            if brace_count == 0:
                # 找到了匹配的闭合大括号，返回内容和它的位置
                return (text[start_pos + 1:pos].strip(), pos)
        pos += 1

    return None


def load_ground_truth_answers(dataset_name: str, starting_index: int = None, end_index: int = None):
    """
    从Hugging Face数据集加载ground truth答案，参考utils.py的实现方式。
    
    Args:
        dataset_name (str): 数据集名称 ('math500' 或 'gsm8k')
        starting_index (int): 起始索引
        end_index (int): 结束索引
        
    Returns:
        tuple: (answers, problems) - 答案列表和问题列表
    """
    try:
        from datasets import load_dataset
        
        if dataset_name == "math500":
            if starting_index is None:
                starting_index = 30
            if end_index is None:
                end_index = 500
            dataset = load_dataset("HuggingFaceH4/MATH-500")
            test_data = dataset["test"]
            answers = test_data["answer"][starting_index:end_index]
            problems = test_data["problem"][starting_index:end_index]
            logging.info(f"成功加载 {len(answers)} 个 math500 答案 (索引 {starting_index}-{end_index-1})")
            
        elif dataset_name == "gsm8k":
            if starting_index is None:
                starting_index = 30
            if end_index is None:
                end_index = 330
            dataset = load_dataset("openai/gsm8k", "main")
            train_data = dataset["train"]
            answers = train_data["answer"][starting_index:end_index]
            problems = train_data["question"][starting_index:end_index]
            logging.info(f"成功加载 {len(answers)} 个 gsm8k 答案 (索引 {starting_index}-{end_index-1})")
            
        else:
            raise ValueError("不支持的数据集。请使用 'math500' 或 'gsm8k'。")
            
        return answers, problems
        
    except Exception as e:
        logging.error(f"加载数据集失败: {e}")
        return None, None


def normalize_answer(answer: str):
    """
    更全面地标准化答案格式以便比较。
    """
    if answer is None:
        return None

    answer = re.sub(r'\\begin\{.*\}|\\end\{.*\}', '', answer)
    answer = re.sub(r'\\text\{.*?\}', '', answer)
    answer = re.sub(r'(?<=\d),(?=\d)', '', answer)
    answer = answer.rstrip('%')
    answer = answer.rstrip('.')
    answer = re.sub(r'\\d?frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', answer)
    answer = re.sub(r'\s*(\$|cm|m|km|kg|g)\s*$', '', answer.strip())

    latex_replacements = {
        r'\\pi': 'pi', r'\\sqrt': 'sqrt', r'\\cdot': '*', r'\\times': '*',
        r'\\div': '/', r'\\pm': '±', r'\\mp': '∓', r'\\infty': 'infinity',
        r'\\left': '', r'\\right': '', r'\\,': ' ', r'\\;': ' ',
        r'\\quad': ' ', r'\\qquad': ' ',
    }

    for latex, replacement in latex_replacements.items():
        answer = answer.replace(latex, replacement)

    answer = re.sub(r'\\+', '', answer)
    answer = answer.replace('{', '').replace('}', '')
    answer = re.sub(r'\s+', '', answer)
    answer = answer.lower()

    return answer


def is_generation_ended_naturally(response_text: str, box_info: tuple | None):
    """
    判断生成是否自然结束。
    只判断回答末尾是否有<eos>，有则代表被截断，未自然结束。
    """
    if response_text.strip().endswith("<eos>"):
        return False
    return True



def analyze_results(results_dir, dataset_name="math500", starting_index=None, end_index=None, alpha_value=100.0):
    """分析TPV intervention结果"""

    results = {
        'total_problems': 0, 'correct': 0, 'answered': 0, 'ended_naturally': 0,
        'details': []
    }

    # 从Hugging Face加载ground truth数据
    ground_truth_answers, ground_truth_problems = load_ground_truth_answers(
        dataset_name, starting_index, end_index
    )
    
    if ground_truth_answers is None:
        logging.error("无法加载ground truth答案，正确性检查将无法进行。")
        return results

    results_path = Path(results_dir)
    if not results_path.exists():
        logging.error(f"结果目录不存在: {results_dir}")
        return results

    problem_dirs = sorted(results_path.glob('problem_*'), key=lambda p: int(p.name.split('_')[1]))

    if not problem_dirs:
        logging.error(f"在 {results_dir} 中没有找到 'problem_*' 格式的目录。")
        return results

    results['total_problems'] = len(problem_dirs)

    for i, problem_dir in enumerate(problem_dirs):
        problem_num = i
        actual_dataset_index = starting_index + i if starting_index is not None else i

        response_file = find_closest_alpha_file(problem_dir, alpha_value)
        if response_file is None:
            logging.warning(f"问题 {problem_num} (目录 {problem_dir.name}) 没有找到任何 response_alpha 文件")
            continue

        try:
            response_text = response_file.read_text(encoding='utf-8')
        except Exception as e:
            logging.error(f"读取文件失败 {response_file}: {e}")
            continue

        # 提取模型答案及其位置
        box_info = extract_boxed_answer(response_text)

        # 根据 box_info 判断是否有答案
        if box_info:
            model_answer, _ = box_info
            # 答案内容不能为空字符串
            has_answer = model_answer.strip() != ""
        else:
            model_answer = None
            has_answer = False

        if has_answer:
            results['answered'] += 1

        # 使用新逻辑判断是否自然结束
        ended_naturally = is_generation_ended_naturally(response_text, box_info)
        if ended_naturally:
            results['ended_naturally'] += 1

        is_correct = False
        gt_answer_extracted = None
        gt_problem_text = None
        
        if ground_truth_answers and problem_num < len(ground_truth_answers):
            gt_text = ground_truth_answers[problem_num]
            gt_problem_text = ground_truth_problems[problem_num] if ground_truth_problems else None
            
            # 注意：ground truth 可能没有box，所以用原始的 extract_boxed_answer 逻辑来提取其内容
            gt_box_info = extract_boxed_answer(str(gt_text))
            if gt_box_info:
                gt_answer_extracted, _ = gt_box_info
            else:
                gt_answer_extracted = str(gt_text).strip()

            if has_answer:
                normalized_model = normalize_answer(model_answer)
                normalized_gt = normalize_answer(gt_answer_extracted)
                is_correct = (normalized_model is not None) and (normalized_model == normalized_gt)

        if is_correct:
            results['correct'] += 1

        results['details'].append({
            'problem_num': problem_num, 
            'dataset_index': actual_dataset_index,
            'has_answer': has_answer, 
            'model_answer': model_answer,
            'is_correct': is_correct, 
            'ground_truth': gt_answer_extracted,
            'ground_truth_problem': gt_problem_text,
            'ended_naturally': ended_naturally, 
            'response_length': len(response_text)
        })

        if (problem_num + 1) % 100 == 0:
            logging.info(f"已处理 {problem_num + 1} / {results['total_problems']} 个问题...")

    return results


def print_summary(results):
    """打印统计摘要"""
    total = results['total_problems']
    if total == 0:
        print("没有找到任何问题进行分析。")
        return

    correct = results['correct']
    answered = results['answered']
    ended = results['ended_naturally']

    print("\n" + "=" * 50)
    print("TPV INTERVENTION 结果统计")
    print("=" * 50)
    print(f"总问题数: {total}")
    print(f"#Correct (正确答案数): {correct} ({correct / total * 100:.1f}%)")
    print(f"#Answered (有答案数): {answered} ({answered / total * 100:.1f}%)")
    print(f"#Ended (自然结束数): {ended} ({ended / total * 100:.1f}%)")
    print("=" * 50)

    if answered > 0:
        accuracy = correct / answered * 100
        print(f"答案准确率 (Correct/Answered): {accuracy:.1f}%")

    print(f"回答率 (Answered/Total): {answered / total * 100:.1f}%")
    print(f"完成率 (Ended/Total): {ended / total * 100:.1f}%")


def save_detailed_results(results, output_file):
    """保存详细结果到JSON文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logging.info(f"详细结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="分析TPV intervention结果")
    parser.add_argument("--results_dir", type=str,
                        default="deepseek_intervention_responses/DeepSeek-R1-Distill-Qwen-32B/math500/intervention",
                        help="结果目录路径")
    parser.add_argument("--dataset", type=str, default="math500", 
                        choices=["math500", "gsm8k"], help="数据集名称")
    parser.add_argument("--starting_index", type=int, default=80, help="数据集起始索引")
    parser.add_argument("--end_index", type=int, default=500, help="数据集结束索引")
    parser.add_argument("--alpha", type=float, default=100.0, help="要分析的alpha值")
    parser.add_argument("--output_json", type=str, default="tpv_analysis_results.json", help="输出详细结果的JSON文件")

    args = parser.parse_args()

    logging.info("开始分析TPV intervention结果...")
    logging.info(f"结果目录: {args.results_dir}")
    logging.info(f"数据集: {args.dataset}")
    logging.info(f"索引范围: {args.starting_index}-{args.end_index}")
    logging.info(f"Alpha值: {args.alpha}")

    results = analyze_results(
        results_dir=args.results_dir,
        dataset_name=args.dataset,
        starting_index=args.starting_index,
        end_index=args.end_index,
        alpha_value=args.alpha
    )

    print_summary(results)

    if args.output_json:
        save_detailed_results(results, args.output_json)

    logging.info("分析完成！")


if __name__ == "__main__":
    main()