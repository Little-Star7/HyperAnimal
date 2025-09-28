import torch
import torch.nn as nn
from omegaconf import OmegaConf
from model import ft_net, ft_net_swinv2, ft_net_convnext, ft_net_dense, ft_net_efficient, ft_net_hr

# ResNet-50
model = ft_net(2000, 0.5, 2, circle=True, ibn=False, linear_num=512)
model.load_state_dict(torch.load("./model/ipanda_gp_f1_resnet50/net_last.pth"))

model_state_dict = model.state_dict()
print(model_state_dict.keys())

model_state_dict = {k.replace("model.", ""): v for k, v in model_state_dict.items()}
model_state_dict = {k: v for k, v in model_state_dict.items() if not k.startswith("classifier.")}
print(model_state_dict.keys())
torch.save(model_state_dict, './pre_model/gp_IDPt_resnet50.pth')

# ConvNeXt
model = ft_net_convnext(2000, 0.5, 2, circle=True, linear_num=512)
model.load_state_dict(torch.load("./model/ipanda_gp_f1_convnext/net_last.pth"))


model_state_dict = model.state_dict()
print(model_state_dict.keys())

model_state_dict = {k.replace("model.", ""): v for k, v in model_state_dict.items()}
model_state_dict = {k: v for k, v in model_state_dict.items() if not k.startswith("classifier.")}
print(model_state_dict.keys())
torch.save(model_state_dict, './pre_model/gp_IDPt_convnext.pth')

#SwinV2
model = ft_net_swinv2(2000, (256, 256), 0.5, 2, circle=True, linear_num=512)
model.load_state_dict(torch.load("./model/ipanda_gp_f1_swinv2/net_last.pth"))


model_state_dict = model.state_dict()
print(model_state_dict.keys())

model_state_dict = {k.replace("model.", ""): v for k, v in model_state_dict.items()}
model_state_dict = {k: v for k, v in model_state_dict.items() if not k.startswith("classifier.")}
print(model_state_dict.keys())
torch.save(model_state_dict, './pre_model/gp_IDPt_swinv2.pth')


# DenseNet121
model = ft_net_dense(2000, 0.5, 2, circle=True, linear_num=512)
model.load_state_dict(torch.load("./model/ipanda_gp_f1_dense/net_last.pth"))

model_state_dict = model.state_dict()
print(model_state_dict.keys())

model_state_dict = {k: v for k, v in model_state_dict.items() if not k.startswith("classifier.")}
model_state_dict = {k.replace("model.", ""): v for k, v in model_state_dict.items()}
print(model_state_dict.keys())
torch.save(model_state_dict, './pre_model/gp_IDPt_dense.pth')

# EfficientNet
model = ft_net_efficient(2000, 0.5, circle=True, linear_num=512)
model.load_state_dict(torch.load("./model/ipanda_gp_f1_efficient/net_last.pth"))


model_state_dict = model.state_dict()
print(model_state_dict.keys())

model_state_dict = {k.replace("model.", ""): v for k, v in model_state_dict.items()}
model_state_dict = {k: v for k, v in model_state_dict.items() if not k.startswith("classifier.")}
print(model_state_dict.keys())
torch.save(model_state_dict, './pre_model/gp_IDPt_efficient.pth')

# HRNet32
model = ft_net_hr(2000, 0.5, circle=True, linear_num=512)
model.load_state_dict(torch.load("./model/ipanda_gp_f1_hr/net_last.pth"))


model_state_dict = model.state_dict()
print(model_state_dict.keys())

model_state_dict = {k.replace("model.", ""): v for k, v in model_state_dict.items()}
model_state_dict = {k: v for k, v in model_state_dict.items() if not k.startswith("classifier.")}
print(model_state_dict.keys())
torch.save(model_state_dict, './pre_model/gp_IDPt_hr.pth')