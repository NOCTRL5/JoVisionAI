import torch
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")
print(torch.cuda.is_available()) 
print(torch.version.cuda)         
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))
