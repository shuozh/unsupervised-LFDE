import numpy as np
import os
from torch.utils.data import Dataset
import cv2
import random
import einops


class HCInewDataset(Dataset):
    """
    return: u v h w c
    """
    def __init__(self, opt):
        super().__init__()

        self.iters_in_one_epoch = opt['iters_in_one_epoch']
        self.patch_size = opt['patch_size']
        self.img_list = self._load_imgs(opt['path'])
        self.traindata_num = len(self.img_list)

    def __getitem__(self, index):
        x = self.img_list[index%self.traindata_num]
        an2, img_h, img_w, _ = x.shape
        start_h = random.randint(0, img_h-self.patch_size)
        start_w = random.randint(0, img_w-self.patch_size)
        train_data = x[:, start_h:start_h+self.patch_size, start_w:start_w+self.patch_size, :]
        train_data = einops.rearrange(train_data, '(u v) h w c -> h w u v c', u=9)
        train_data = train_data[:, :, 1:-1, 1:-1, :]
        train_data = einops.rearrange(train_data, 'h w u v c -> (u h) (v w) c', u=7)
        if random.random() < 0.5:
            train_data = train_data[::-1, ...]
        if random.random() < 0.5:
            train_data = train_data[:, ::-1, :]
        if random.random() < 0.5:
            train_data = train_data.transpose(1, 0, 2)
        orientation_rand = random.randint(0, 4)
        if orientation_rand == 0:
            train_data = np.rot90(train_data, 1)
        if orientation_rand == 1:
            train_data = np.rot90(train_data, 2)
        if orientation_rand == 2:
            train_data = np.rot90(train_data, 3)
        train_data = einops.rearrange(train_data, '(u h) (v w) c ->u v h w c', u=7, v=7)
        train_data = train_data.copy()

        return train_data
        
    def __len__(self):
        return self.iters_in_one_epoch

    def _load_imgs(self, dataset_path):
        
        dataset_paths = [
            f'{dataset_path}/additional'
        ]
        img_list = []
        for dataset_path in dataset_paths:
            name_list = os.listdir(dataset_path)
            # name_list = os.listdir(dataset_path)[:3]
            name_list = [
                # 'backgammon', 
                # 'dots',
                # 'pyramids', 
                # 'stripes',
                # 'boxes',
                # 'cotton',
                # 'dino', 
                'sideboard', 
                # 'antinous',
                # 'boardgames',
                # 'dishes',
                # 'greek',
                # 'medieval2',
                # 'pens',
                # 'pillows', 
                # 'platonic',
                # 'rosemary',
                # 'table',
                # 'tomb',
                # 'tower',
                # 'town',
                ]            
            for name in name_list:
                if name not in ['kitchen', 'museum', 'vinyl']:
                    print(name)
                    lf_list = []
                    for i in range(81):
                        tmp = cv2.imread(f'{dataset_path}/{name}/input_Cam{i:03}.png')  # load LF images(9x9)
                        lf_list.append(tmp)
                        img = np.stack(lf_list, 0) # n h w c
                    # img = cv2.imread(f'{dataset_path}/{name}')
                    img = img/255
                    img = np.float32(img)
                    img_list.append(img)
            
        return img_list

class HCInewDatasetMask(Dataset):
    """
    return: u v h w c
    """
    def __init__(self, opt):
        super().__init__()

        self.iters_in_one_epoch = opt['iters_in_one_epoch']
        self.patch_size = opt['patch_size']
        name_list = [
            # 'backgammon', 
            # 'dots',
            # 'pyramids', 
            # 'stripes',
            # 'boxes',
            # 'cotton',
            # 'dino', 
            'sideboard', 
            # 'antinous',
            # 'boardgames',
            # 'dishes',
            # 'greek',
            # 'medieval2',
            # 'pens',
            # 'pillows', 
            # 'platonic',
            # 'rosemary',
            # 'table',
            # 'tomb',
            # 'tower',
            # 'town',
            ]
        self.img_list = self._load_imgs(opt['path'], name_list)
        self.traindata_num = len(self.img_list)
        self.mask_list = self._load_masks(name_list, 0.2)
        self.name_list = name_list

    def __getitem__(self, index):
        scene_id = index%self.traindata_num
        x = self.img_list[scene_id]
        mask = self.mask_list[scene_id]
        an2, img_h, img_w, _ = x.shape

        start_h = random.randint(0, img_h-self.patch_size)
        start_w = random.randint(0, img_w-self.patch_size)
        train_data = x[:, start_h:start_h+self.patch_size, start_w:start_w+self.patch_size, :]
        mask_patch = mask[start_h:start_h+self.patch_size, start_w:start_w+self.patch_size]
        train_data = einops.rearrange(train_data, '(u v) h w c -> h w u v c', u=9)
        train_data = train_data[:, :, 1:-1, 1:-1, :]
        train_data = einops.rearrange(train_data, 'h w u v c -> (u h) (v w) c', u=7)
        if random.random() < 0.5:
            train_data = train_data[::-1, ...]
            mask_patch = mask_patch[::-1, ...]
        if random.random() < 0.5:
            train_data = train_data[:, ::-1, :]
            mask_patch = mask_patch[:, ::-1]
        if random.random() < 0.5:
            train_data = train_data.transpose(1, 0, 2)
            mask_patch = mask_patch.transpose(1, 0)
        orientation_rand = random.randint(0, 4)
        if orientation_rand == 0:
            train_data = np.rot90(train_data, 1)
            mask_patch = np.rot90(mask_patch, 1)
        if orientation_rand == 1:
            train_data = np.rot90(train_data, 2)
            mask_patch = np.rot90(mask_patch, 2)
        if orientation_rand == 2:
            train_data = np.rot90(train_data, 3)
            mask_patch = np.rot90(mask_patch, 3)
        train_data = einops.rearrange(train_data, '(u h) (v w) c ->u v h w c', u=7, v=7)
        train_data = train_data.copy()
        mask_patch = mask_patch.copy()

        return train_data, mask_patch
        
    def __len__(self):
        return self.iters_in_one_epoch

    def _load_imgs(self, dataset_path, name_list):
        
        dataset_paths = [
            f'{dataset_path}/additional'
        ]
        img_list = []
        for dataset_path in dataset_paths:
            if name_list == None:
                name_list = os.listdir(dataset_path)
            for name in name_list:
                if name not in ['kitchen', 'museum', 'vinyl']:
                    print(name)
                    lf_list = []
                    for i in range(81):
                        tmp = cv2.imread(f'{dataset_path}/{name}/input_Cam{i:03}.png')  # load LF images(9x9)
                        lf_list.append(tmp)
                        img = np.stack(lf_list, 0) # n h w c
                    img = img/255
                    img = np.float32(img)
                    img_list.append(img)

        return img_list

    def _load_masks(self, name_list, th=0.1):
        nlist = []
        numlist = []
        for img in name_list:
            num = np.load(f'../log/hci/occ_num/{th}/{img}.npy')
            nlist.append(num)
            numlist.append(np.average((num>5)*1))
        print(np.average(numlist))  # 计算掩码率
        return nlist

class RealDataset(Dataset):
    """
    return: u v h w c
    """
    def __init__(self, opt):
        super().__init__()

        self.iters_in_one_epoch = opt['iters_in_one_epoch']
        self.patch_size = opt['patch_size']
        self.img_list = self._load_imgs(opt['path'])
        self.traindata_num = len(self.img_list)

    def __getitem__(self, index):
        x = self.img_list[index%self.traindata_num]
        u, v, img_h, img_w, _ = x.shape
        # print(x.shape)
        while True:
            valid = 1
            start_h = random.randint(40, img_h-self.patch_size)
            start_w = random.randint(40, img_w-self.patch_size)
            # print(start_h, start_w)
            train_data = x[:, :, start_h:start_h+self.patch_size, start_w:start_w+self.patch_size, :]
            center_pixel = train_data[3, 3, self.patch_size//2, self.patch_size//2]
            sum_diff = np.average(np.abs(train_data[3, 3, :, :,] - center_pixel))
            # print(sum_diff)
            if sum_diff < 0.01:
                valid = 0
            if valid == 1:        
                train_data = einops.rearrange(train_data, 'u v h w c -> (u h) (v w) c', u=7)

                if random.random() < 0.5:
                    train_data = train_data[::-1, ...]
                if random.random() < 0.5:
                    train_data = train_data[:, ::-1, :]
                if random.random() < 0.5:
                    train_data = train_data.transpose(1, 0, 2)
                orientation_rand = random.randint(0, 4)
                if orientation_rand == 0:
                    train_data = np.rot90(train_data, 1)
                if orientation_rand == 1:
                    train_data = np.rot90(train_data, 2)
                if orientation_rand == 2:
                    train_data = np.rot90(train_data, 3)
                train_data = einops.rearrange(train_data, '(u h) (v w) c ->u v h w c', u=7, v=7)
                train_data = train_data.copy()

                return train_data
        
    def __len__(self):
        return self.iters_in_one_epoch

    def _load_imgs(self, dataset_path):
        # names = os.listdir(dataset_path)
        names = [
            'IMG_1324_eslf.npy',
            'IMG_1340_eslf.npy',
            'IMG_1328_eslf.npy',
            ]
        img_list = []
        for name in names:
            print(name)
            img = np.load(f'{dataset_path}/{name}')
            img = img/255
            img = np.float32(img)
            img_list.append(img)
            
        return img_list

class RealDatasetMask(Dataset):
    """
    return: u v h w c
    """
    def __init__(self, opt):
        super().__init__()

        self.iters_in_one_epoch = opt['iters_in_one_epoch']
        self.patch_size = opt['patch_size']
        self.names = [
            'Rock',
            'Flower1',
            'Flower2',
            ]
        self.img_list = self._load_imgs(opt['path'])
        self.mask_list = self._load_masks()
        self.traindata_num = len(self.img_list)

    def __getitem__(self, index):
        x = self.img_list[index%self.traindata_num]
        mask = self.mask_list[index%self.traindata_num]
        u, v, img_h, img_w, _ = x.shape
        # print(x.shape)
        while True:
            valid = 1
            start_h = random.randint(40, img_h-self.patch_size)
            start_w = random.randint(40, img_w-self.patch_size)
            # print(start_h, start_w)
            train_data = x[:, :, start_h:start_h+self.patch_size, start_w:start_w+self.patch_size, :]
            mask_patch = mask[start_h:start_h+self.patch_size, start_w:start_w+self.patch_size]
            center_pixel = train_data[3, 3, self.patch_size//2, self.patch_size//2]
            sum_diff = np.average(np.abs(train_data[3, 3, :, :,] - center_pixel))
            # print(sum_diff)
            if sum_diff < 0.01:
                valid = 0
            if valid == 1:        
                train_data = einops.rearrange(train_data, 'u v h w c -> (u h) (v w) c', u=7)

                if random.random() < 0.5:
                    train_data = train_data[::-1, ...]
                    mask_patch = mask_patch[::-1, ...]
                if random.random() < 0.5:
                    train_data = train_data[:, ::-1, :]
                    mask_patch = mask_patch[:, ::-1]
                if random.random() < 0.5:
                    train_data = train_data.transpose(1, 0, 2)
                    mask_patch = mask_patch.transpose(1, 0)
                orientation_rand = random.randint(0, 4)
                if orientation_rand == 0:
                    train_data = np.rot90(train_data, 1)
                    mask_patch = np.rot90(mask_patch, 1)
                if orientation_rand == 1:
                    train_data = np.rot90(train_data, 2)
                    mask_patch = np.rot90(mask_patch, 2)
                if orientation_rand == 2:
                    train_data = np.rot90(train_data, 3)
                    mask_patch = np.rot90(mask_patch, 3)
                train_data = einops.rearrange(train_data, '(u h) (v w) c ->u v h w c', u=7, v=7)
                train_data = train_data.copy()
                mask_patch = mask_patch.copy()

                return train_data, mask_patch
        
    def __len__(self):
        return self.iters_in_one_epoch

    def _load_imgs(self, dataset_path):
        # names = os.listdir(dataset_path)
        names = self.names
        img_list = []
        for name in names:
            print(name)
            img = np.load(f'{dataset_path}/{name}.npy')
            img = img/255
            img = np.float32(img)
            img_list.append(img)
            
        return img_list
    
    def _load_masks(self):
        mask_path= '../log/img/mask'
        names = self.names
        mask_list = []
        for name in names:
            mask = np.load(f'{mask_path}/{name}/occ_mask.npy')
            mask_list.append(mask)
        return mask_list
    

# if __name__ == "__main__":
#     dataset = RealDataset()
