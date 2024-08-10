import torch 
import torch.nn as nn
import torch.nn.utils.parametrize as P


class DigitClassification(nn.Module):
    def __init__(self, input_size, output_size, features = None):
        super(DigitClassification,self).__init__()
        
        self.input_size = input_size
        self.features = features
        if self.features is None:
            self.features = [input_size, 512, 1000, 2000]
        
        ListModules = []
        for ii in range(len(self.features) - 1):
            feature = self.features[ii]
            feature_next = self.features[ii + 1]
            ListModules.append(nn.Linear(feature, feature_next))
        ListModules.append(nn.Linear(self.features[-1], output_size))
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.layers = nn.ModuleList(ListModules)
    
    def forward(self, x):
        out = x.view(-1, self.input_size).clone()
        n = len(self.layers)
        for ii, layer in enumerate(self.layers):
            out = layer(out)
            if ii < n - 1:
                out = self.relu(out)
            else:
                out = out#self.sigmoid(out)        
        return out

# The following code is used to parametrise the layer
class LoRaParametrisation(nn.Module):
    def __init__(self, in_channel, out_channel, rank = 2, alpha = 1, device = "cpu"):
        super(LoRaParametrisation, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.rank = rank
        self.device = device
        self.alpha = alpha
        self.lora_A = nn.Parameter(torch.zeros(self.in_channel, self.rank, device = self.device))
        self.lora_B = nn.Parameter(torch.zeros(self.rank, self.out_channel, device = self.device))
        
        nn.init.normal_(self.lora_A, mean = 0, std = 1)
        nn.init.zeros_(self.lora_B)
        
        self.scale = self.alpha / self.rank
        self.enabled = True
        
    def forward(self, original_weights):
        if self.enabled:
            return original_weights + torch.matmul(self.lora_A, self.lora_B).view(original_weights.shape) * self.scale
        return original_weights

# The following code is used to parametrise the layer   
def layer_parametrisation(layer, rank = 2, device = "cpu"):
    in_channel, out_channel = layer.weight.shape
    return LoRaParametrisation(in_channel, out_channel, rank = rank, device = device)

# The following code is used to apply LoRa to the model
def LoRa_model(model, rank = 2, device = "cpu"):
    # apply LoRa to the model
    for layer in model.layers:
        if isinstance(layer, nn.Linear):
            P.register_parametrization(layer, "weight", layer_parametrisation(layer, rank = rank, device = device))
    
    # freeze non-LoRA parameters
    num = 0
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False
            num += 1
    print(f'Number of Layers frozen: {num}')
    
    total_parameters_non_lora = 0
    total_parameters_lora = 0
    for index, layer in enumerate(model.layers):
        total_parameters_lora += layer.parametrizations["weight"][0].lora_A.nelement() + layer.parametrizations["weight"][0].lora_B.nelement()
        total_parameters_non_lora += layer.weight.nelement() + layer.bias.nelement()
        print(
            f'Layer {index+1} -- Original Weights: {layer.weight.nelement()}  Bias: {layer.bias.nelement()} ---- Lora_A weight: {layer.parametrizations["weight"][0].lora_A.nelement()} + Lora_B weight: {layer.parametrizations["weight"][0].lora_B.nelement()}'
        )
        
    print(f"Total trainable parameters of the model: {total_parameters_non_lora} (non-LoRa) vs {total_parameters_lora} (LoRa) Ratio: {(total_parameters_lora/total_parameters_non_lora)*100:.2f}% of the original model")
    return model

# The following code is used to enable or disable the LoRa parameters in the model
def enable_disable_lora(model, enabled=True):
    for layer in model.layers:
        for name, _ in layer.named_parameters():
            if "lora_" in name:
                layer.parametrizations["weight"][0].enabled = enabled
                

# The following code is used to count the number of parameters in the model  
def count_parameters(model):
    tparas = 0
    for layer in model.layers:
        tparas += layer.weight.nelement() + layer.bias.nelement()
    return tparas