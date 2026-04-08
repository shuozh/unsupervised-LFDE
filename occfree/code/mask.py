import numpy as np
import cv2


def calmask_hcinew():
    name_list = ['']
    for name in name_list:
        disp_list = []
        for i in range(1, 43):
            name_dir = f'../log/img/UNetRGB_ULossTopkPre/{i}/{name}'
            disp = np.load(name_dir)
            disp_list.append(disp)
        disp = np.stack(disp_list, 2)
        base = np.average(disp[:, :, -6:], 2)
        mask = np.sum((np.abs(disp - base)>0.1)*1, 2)
        np.save(f'../log/mask/{name}', mask)
        cv2.imwrite(f'../log/mask/{name}.png', np.uint8(mask *5))