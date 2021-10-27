import torch

def compareModelWeights(model_a, model_b):
    module_a = model_a._modules
    module_b = model_b._modules
    if len(list(module_a.keys())) != len(list(module_b.keys())):
        return False
    a_modules_names = list(module_a.keys())
    b_modules_names = list(module_b.keys())
    sum_diff = 0
    for i in range(len(a_modules_names)):
        layer_name_a = a_modules_names[i]
        layer_name_b = b_modules_names[i]
        layer_a = module_a[layer_name_a]
        layer_b = module_b[layer_name_b]
        if hasattr(layer_a, 'weight') and hasattr(layer_b, 'weight'):
            sum_diff += abs(mean(layer_a.weight.data-layer_b.weight.data)):
    return sum_diff