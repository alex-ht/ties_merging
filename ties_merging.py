import torch
from transformers import AutoModelForImageTextToText
from typing import List


def keep_topk_reset_rest_to_zero(tensors, k=20):
    """
    First, flatten the tensors and concatenate them to find the threshold value that distinguishes the top k% elements. 
    The top k% calculation considers only the absolute values, ignoring the sign.
    After obtaining the threshold value, clip each tensor by setting elements with absolute values smaller than the threshold to 0, 
    while keeping elements with absolute values greater than or equal to the threshold.

    Args:
        tensors: A list of tensors.
        k: The percentage of top elements to keep (e.g., k=20 means keeping the top 20%).

    Returns:
        A list of tensors with only the top k% elements kept.
    """
    flattened = torch.cat([t.flatten() for t in tensors])
    num_elements = flattened.numel()
    topk_count = max(1, int(num_elements * (k / 100)))  # Calculate the number of elements corresponding to the top k%
    topk_values, _ = torch.topk(flattened.abs(), topk_count)
    threshold = topk_values[-1]  # The threshold value for the top k% absolute values

    clipped_tensors = []
    for tensor in tensors:
        clipped_tensor = tensor.clone()
        clipped_tensor[clipped_tensor.abs() < threshold] = 0
        clipped_tensors.append(clipped_tensor)

    return clipped_tensors
    
def ties_merging(models: List[AutoModelForImageTextToText], reference_model: AutoModelForImageTextToText, device="cpu") -> AutoModelForImageTextToText:
    """
    Merges multiple models using the TIES-MERGING algorithm with respect to a reference model.

    Args:
        models: A list of AutoModelForImageTextToText models to merge.
        reference_model: The reference AutoModelForImageTextToText model.
        device: The device to use for computation.

    Returns:
        A merged AutoModelForImageTextToText model.
    """
    # 1. Calculate task vectors (difference from the reference model)
    with torch.no_grad():
        task_vectors = []
        for model in models:
            task_vector = {}
            for name, param in model.named_parameters():
                reference_param = reference_model.get_parameter(name)
                task_vector[name] = (param.data - reference_param.data)
            task_vectors.append(task_vector)

        # 2. Trim redundant parameters
        trimed_task_vectors = []
        for task_vector in task_vectors:
            # Use keep_topk_reset_rest_to_zero to keep only the top k% of parameters
            trimmed_task_vector = {}
            for name, param in task_vector.items():
                trimmed_param = keep_topk_reset_rest_to_zero([param], k=20)[0]
                trimmed_task_vector[name] = trimmed_param
            trimed_task_vectors.append(trimmed_task_vector)
        del task_vectors  # Free memory
        
        # 3. Find the most common sign for each parameter across trimed_task_vectors
        for name in trimed_task_vectors[0].keys():
            signs = torch.stack([torch.sign(task_vector[name]) for task_vector in trimed_task_vectors])
            majority_sign = torch.sign(signs.sum(dim=0))  # Majority sign across models

            for task_vector in trimed_task_vectors:
                inconsistent_mask = torch.sign(task_vector[name]) != majority_sign
                task_vector[inconsistent_mask] = 0  # Set inconsistent parameters to zero

        # 4. Merge the trimmed task vectors into a new task vector by averaging non-zero elements.
        merged_task_vector = {}
        for name in trimed_task_vectors[0].keys():
            params = torch.stack([task_vector[name] for task_vector in trimed_task_vectors])
            non_zero_mask = params != 0
            non_zero_count = non_zero_mask.sum(dim=0).float()
            non_zero_count[non_zero_count == 0] = 1  # Avoid division by zero
            merged_param = params.sum(dim=0) / non_zero_count
            merged_task_vector[name] = merged_param

        # 5. Apply the merged task vector to the reference model to create the merged model
        merged_model = reference_model
        for name, param in merged_model.named_parameters():
            param += merged_task_vector[name]
        
    return merged_model

def load_models(model_names: List[str], device="cpu") -> List[AutoModelForImageTextToText]:
    """
    Loads multiple models from the Hugging Face Model Hub.

    Args:
        model_names: A list of model names to load.
        device: The device to use for the models.

    Returns:
        A list of AutoModelForImageTextToText models.
    """
    models = []
    for model_name in model_names:
        model = AutoModelForImageTextToText.from_pretrained(model_name).to(device)
        models.append(model)
    return models

def merge_and_save(model_names: List[str], reference_model_name: str, output_path: str, device="cpu"):
    """
    Loads, merges, and saves the merged model.

    Args:
        model_names: A list of model names to load and merge.
        reference_model_name: The name of the reference model.
        output_path: The path to save the merged model.
        device: The device to use for computation.
    """
    models = load_models(model_names, device)
    reference_model = AutoModelForImageTextToText.from_pretrained(reference_model_name).to(device)
    merged_model = ties_merging(models, reference_model, device)
    merged_model.save_pretrained(output_path)
    print(f"Merged model saved to {output_path}")

if __name__ == '__main__':
    # Example usage:
    model_names = [
        "alexredna/TinyLlama-1.1B-Chat-v1.0-reasoning-sft-full",
        "yzhuang/TinyLlama-1.1B_fictional",
    ]
    reference_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    output_path = "merged_model"
    merge_and_save(model_names, reference_model_name, output_path)
