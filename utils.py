# This software is for non-commercial use only.
# Commercial use requires a separate license.

from datasets import load_dataset


def load_problems(dataset, starting_index=None, end_index=None):
    """
    Load a specified number of problems from a dataset starting from a given index.
    
    Args:
        dataset (str): The name of the dataset to load.
        starting_index (int): The index to start loading problems from.
        num_problems (int): The number of problems to load.
        
    Returns:
        list: A list of problems loaded from the dataset.
    """
    # Load the dataset
    if dataset == "math500":
        if starting_index is None:
            starting_index = 30
        if end_index is None:
            end_index = 500
        problems = load_dataset("HuggingFaceH4/MATH-500")["test"]["problem"][starting_index:end_index]
    
    elif dataset == "gsm8k":
        if starting_index is None:
            starting_index = 30
        if end_index is None:
            end_index = 330
        problems = load_dataset("openai/gsm8k", "main")["train"]["question"][starting_index:end_index]

    elif dataset =="svamp":
        if starting_index is None:
            starting_index = 30
        if end_index is None:
            end_index = 330
        problems = load_dataset("ChilleD/SVAMP", "default")["train"]["question_concat"][starting_index:end_index]

    
    return problems