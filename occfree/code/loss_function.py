import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils import warp_all
from einops import rearrange
import cv2

def img_grads(I):  # shape: b c h w
    I_dy = I[:, :, 1:, :] - I[:, :, :-1, :]
    I_dx = I[:, :, :, 1:] - I[:, :, :, :-1]
    return I_dx, I_dy

def Edge_Aware_Smoothness_LossRGB(D, I, edge_constant=150):

    img_gx_r, img_gy_r = img_grads(I[:, 0:1, ...])
    img_gx_g, img_gy_g = img_grads(I[:, 1:2, ...])
    img_gx_b, img_gy_b = img_grads(I[:, 2:3, ...])

    weight_x = torch.exp(-edge_constant * (torch.abs(img_gx_r)+torch.abs(img_gx_g)+torch.abs(img_gx_b))/3.)
    weight_y = torch.exp(-edge_constant * (torch.abs(img_gy_r)+torch.abs(img_gy_g)+torch.abs(img_gy_b))/3.)
    D = D.unsqueeze(dim=1)
    disp_gx, disp_gy = img_grads(D)
    
    loss = (torch.mean((weight_x * torch.abs(disp_gx))[:, :, 8:-8, 8:-8]) + torch.mean((weight_y * torch.abs(disp_gy))[:, :, 8:-8, 8:-8]))/2.
    return loss

def Edge_Aware_Smoothness_LossRGB_Mask(D, I, edge_constant=150):

    img_gx_r, img_gy_r = img_grads(I[:, 0:1, ...])
    img_gx_g, img_gy_g = img_grads(I[:, 1:2, ...])
    img_gx_b, img_gy_b = img_grads(I[:, 2:3, ...])

    weight_x = torch.exp(-edge_constant * (torch.abs(img_gx_r)+torch.abs(img_gx_g)+torch.abs(img_gx_b))/3.)
    weight_y = torch.exp(-edge_constant * (torch.abs(img_gy_r)+torch.abs(img_gy_g)+torch.abs(img_gy_b))/3.)
    D = D.unsqueeze(dim=1)
    disp_gx, disp_gy = img_grads(D)
    
    loss = (torch.mean((weight_x * torch.abs(disp_gx))[:, :, 8:-8, 8:-8].unsqueeze(1).unsqueeze(1).unsqueeze(1)) + torch.mean((weight_y * torch.abs(disp_gy))[:, :, 8:-8, 8:-8].unsqueeze(1).unsqueeze(1).unsqueeze(1)))/2.
    return loss


class ULossTopkPre(nn.Module):
    def __init__(self):
        '''alpha: 梯度损失所占比例，0为不计算梯度损失
        选择损失小的一半视角'''
        super().__init__()
    def forward(self, pred, x, epoch):
        if epoch < 200:
            k = 0
        elif epoch < 2300:
            k = ((epoch-200) // 100)*2
        else:
            k = 44
        # k = 45
        alpha = 0.1

        device = x.get_device()
        x = rearrange(x, 'b u v h w c -> b c (u v) h w')
        warped = warp_all(pred, x, device, 7)
        center = x[:, :, 24:25, :, :]
        # warped = warped / (torch.mean(warped, dim=2, keepdim=True)+1e-8)
        warped_center = warped[:, :, 24:25, :, :]
        gradient_loss = Edge_Aware_Smoothness_LossRGB(pred, center.squeeze(dim=2))  # 梯度损失
        color_loss = torch.abs((warped - warped_center) * 1)  # b 1 81 h w
        topk, topk_index = torch.topk(color_loss, 49-k, dim=2, largest=False)
        mask = torch.zeros_like(color_loss)
        mask = mask.scatter_(2, topk_index, 1, reduce='add')
        color_loss = color_loss * mask *(49/(49-k))
        # color_loss = torch.clip(color_loss, 0, 0.1)
        color_loss = color_loss[..., 8:-8, 8:-8]
        color_loss = torch.mean(color_loss)  
        # print(color_loss, gradient_loss)
        # 根据预测视差反向warp后 与原光场图像的差的绝对值
        loss = color_loss + gradient_loss*alpha
        return loss

class ULossRGBTopakgnc(nn.Module):
    def __init__(self):
        '''alpha: 梯度损失所占比例，0为不计算梯度损失
        选择损失小的一半视角'''
        super().__init__()
        kernel = [[0.0751, 0.1238, 0.0751],
        [0.1238, 0.2042, 0.1238],
        [0.0751, 0.1238, 0.0751],]
        # kernel = [[0.0113, 0.0838, 0.0113],
        #         [0.0838, 0.6193, 0.0838],
        #         [0.0113, 0.0838, 0.0113]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = torch.nn.Parameter(data=kernel, requires_grad=False)
        self.alpha = 0.1
    def forward(self, pred, x, y, epoch):
        device = x.get_device()
        alpha = self.alpha
        x = rearrange(x, 'b u v h w c -> b c (u v) h w')
        center = x[:, :, 24:25, :, :]
        gradient_loss = Edge_Aware_Smoothness_LossRGB_Mask(pred, center.squeeze(dim=2))  # 梯度损失
        warped = warp_all(pred, x, device, 7)
        # warped = warped / (torch.mean(warped, dim=2, keepdim=True)+1e-8)
        color_loss = torch.abs(warped - warped[:, :, 24:25, ...])  # b 1 81 h w
        color_loss = torch.mean(color_loss, 1, keepdim=True)
        b, c, n, h, w = color_loss.shape
        if epoch > 0:
            with torch.no_grad():
                color_loss_gauss = rearrange(color_loss, 'b c (n1 n2) h w -> (b h w) c n1 n2', n1=7)
                color_loss_gauss = F.pad(color_loss_gauss, ((1, 1, 1, 1)), mode='replicate')
                weight = self.weight.to(device)
                color_loss_gauss = F.conv2d(color_loss_gauss, weight)
                color_loss_gauss = rearrange(color_loss_gauss, '(b h w) c n1 n2  -> b c (n1 n2) h w', h=h, w=w)
                sorted_gauss, sorted_gauss_index = torch.sort(color_loss_gauss, 2, descending=True)
                
                mask = torch.arange(0, n).unsqueeze(1).unsqueeze(1).unsqueeze(0).unsqueeze(0).expand_as(color_loss_gauss)
                y = y.unsqueeze(1).unsqueeze(1)
                mask = mask.to(device).float()
                mask = (mask > y)/(n-y)*n # y表示遮挡的数量
            sorted = torch.take_along_dim(color_loss, sorted_gauss_index, 2)
        else:
            sorted, _ = torch.sort(color_loss, 2, descending=True)
            mask = torch.arange(0, n).unsqueeze(1).unsqueeze(1).unsqueeze(0).unsqueeze(0).expand_as(color_loss)
            y = y.unsqueeze(1).unsqueeze(1)
            mask = mask.to(device).float()
            mask = (mask > y)/(n-y)*n # y表示遮挡的数量
        color_loss = (sorted * mask)
        color_loss = color_loss[..., 8:-8, 8:-8]
        color_loss = torch.mean(color_loss) 
        loss = color_loss + gradient_loss*alpha
        return loss

