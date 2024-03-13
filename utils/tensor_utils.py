import torch

def normalize_tensor(tensor):
    mean = torch.mean(tensor)
    std = torch.std(tensor)
    normalized_tensor = (tensor - mean) / std
    return normalized_tensor

def change_tensor(tensor,target_size):
    output_tensor = torch.zeros(target_size)
    output_tensor[:tensor.size(0), :, :] = tensor
    return output_tensor