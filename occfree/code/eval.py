import cv2
import numpy as np
import torch
from einops import rearrange
import os
import model as mymodel
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import utils

def eval_HCInew(checkpoint_path, device, model, resultdir):

    model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    dataset_path = "../../../dataset/hci_dataset/additional"
    # name_list = os.listdir(dataset_path)
    name_list = [
            # 'backgammon', 
            # 'dots',
            # 'pyramids', 
            # 'stripes',
            # 'boxes',
            # 'cotton',
            # 'dino', 
            'sideboard', 
            # 'museum',
            # 'kitchen',
            # 'vinyl'
            # 'pens'
                 ]
    model.eval()
    if not os.path.exists(f'../log/img/{resultdir}'):
        os.makedirs(f'../log/img/{resultdir}')
    for name in name_list:
        lf_list = []
        depth_lable = utils.read_pfm(f'{dataset_path}/{name}/gt_disp_lowres.pfm')
        # np.save(f'../log/img/{resultdir}/{name}_gt.npy', depth_lable)
        for i in range(81):
            tmp = cv2.imread(f'{dataset_path}/{name}/input_Cam{i:03}.png')  # load LF images(9x9)
            lf_list.append(tmp)
            img_read = np.stack(lf_list, 0) # n h w c   
        # img_read = img_read[:, :128, :128, :]
        img_read = rearrange(img_read, '(u v) h w c -> u v h w c', u=9)    
        img_read = img_read[1:-1, 1:-1, ...] 
        img = img_read/255
        img = np.float32(img)
        eval_data = img[np.newaxis, ...]

        eval_data = torch.from_numpy(eval_data).to(device).float()
        with torch.no_grad():
            out = model(eval_data)
        out = out.cpu().numpy()[0, ...]
        print(name)
        print(np.average(np.square(out - depth_lable)[..., 10:-10, 10:-10]))
        print(np.average(np.abs((out - depth_lable)[..., 10:-10, 10:-10]>0.07)*1))

        img_save = np.uint8((out+4)*255/8)
        cv2.imwrite(f'../log/img/{resultdir}/{name}.png', img_save)
        np.save(f'../log/img/{resultdir}/{name}', out)

def eval_Real(checkpoint_path, device, model, resultdir):

    model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    dataset_path = "../../../dataset/Real/DenoiseTrain"
    # name_list = os.listdir(dataset_path)
    name_list = [
        'occlusions_36_eslf.npy',
        'occlusions_37_eslf.npy',
                 ]
    model.eval()
    if not os.path.exists(f'../log/img/{resultdir}'):
        os.makedirs(f'../log/img/{resultdir}')
    for name in name_list:
        print(name)
        img_read = np.load(f'{dataset_path}/{name}')  # load LF images(9x9)
        # img_read = img_read[:, :128, :128, :]
        img = img_read/255
        img = np.float32(img)
        gauss = np.random.normal(0.0, 0.0, img.shape)
        eval_data = img +gauss
        eval_data_gauss = eval_data
        eval_data = eval_data[np.newaxis, ...]

        eval_data = torch.from_numpy(eval_data).to(device).float()
        with torch.no_grad():
            out = model(eval_data)
        out = out.cpu().numpy()[0, ...]
        np.save(f'../log/img/{resultdir}/{name}', out)
        # print(out)
        img_save = np.uint8((out+2)*255/4)
        cv2.imwrite(f'../log/img/{resultdir}/{name[:-4]}.png', img_save)

def eval_HCInewPre(device, model, resultdir):
    resultdir = 'UNetRGB_ULossTopkPre'
    model.to(device)
    ckp_path = '../log/ckp/UNetRGB_ULossTopkPre/'
    k_name = os.listdir(ckp_path)
    dataset_path = "../../../dataset/hci_dataset/additional"
    name_list = os.listdir(dataset_path)

    for k in k_name:
        print(k)
        checkpoint = torch.load(f'{ckp_path}{k}', map_location=device)
        model.load_state_dict(checkpoint['model'])
        
        model.eval()
        kk = int(int(k.split("_")[0])/50)
        if not os.path.exists(f'../log/img/{resultdir}/{kk}'):
            os.makedirs(f'../log/img/{resultdir}/{kk}')
        for name in name_list:
            print(name)
            lf_list = []
            depth_lable = utils.read_pfm(f'{dataset_path}/{name}/gt_disp_lowres.pfm')
            for i in range(81):
                tmp = cv2.imread(f'{dataset_path}/{name}/input_Cam{i:03}.png')  # load LF images(9x9)
                lf_list.append(tmp)
                img_read = np.stack(lf_list, 0) # n h w c   
            img_read = rearrange(img_read, '(u v) h w c -> u v h w c', u=9)    
            img_read = img_read[1:-1, 1:-1, ...] 
            img = img_read/255
            eval_data = np.float32(img)
            eval_data = eval_data[np.newaxis, ...]
            eval_data = torch.from_numpy(eval_data).to(device).float()
            with torch.no_grad():
                out = model(eval_data)
            out = out.cpu().numpy()[0, ...]
            print(np.average(np.square(out - depth_lable)[10:-10, 10:-10]))
            np.save(f'../log/img/{resultdir}/{kk}/{name}.npy', out)
            img_save = np.uint8((out+4)*255/8)
            cv2.imwrite(f'../log/img/{resultdir}/{kk}/{name}.png', img_save)





if __name__ == "__main__":

    model_name = 'UNetRGB'
    ckp = f'../log/ckp/UNetRGB/unethci.pth'
    device = torch.device('cuda:0')
    model = getattr(mymodel, model_name)(None)
    eval_HCInew(ckp, device, model, model_name)




