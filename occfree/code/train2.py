import argparse
from torch.utils.data import DataLoader
import utils
import torch
import torch.nn as nn
import datetime
import os

def train(opt):
    train_dataset_opt = opt['train_dataset']
    ckp_dir = f'../log/ckp/{opt["model"]}_{opt["loss_name"]}_{train_dataset_opt["type"]}_finetune'
    if not os.path.exists(ckp_dir):
        os.makedirs(ckp_dir)
    TrainDataset = getattr(__import__('dataloader'), train_dataset_opt['type'])
    train_set = TrainDataset(train_dataset_opt['args'])
    train_loader = DataLoader(train_set, batch_size=train_dataset_opt['batch_size'], shuffle=True,
                              num_workers=0, drop_last=True)
    device = torch.device(opt['device'])
    Model = getattr(__import__('model'), opt['model'])
    model = Model(opt['model_setting'])
    model.to(device)

    criteron = getattr(__import__('loss_function'), opt['loss_name'])()
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt['lr'])
    epochs = opt['epochs']
    for epoch in range(epochs):
        time1 = datetime.datetime.now()
        train_loss = 0.0
        batch_num = 0
        for i, [x, y] in enumerate(train_loader):
            x = x.to(device).float()
            y = y.to(device).float()
            output = model(x)
            optimizer.zero_grad()
            loss = criteron(output, x, y, epoch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            batch_num += 1

        train_loss = train_loss/batch_num
        time3 = datetime.datetime.now()
        print(f'epoch{epoch} finished, loss = {train_loss}, time cost = {time3 - time1}')
        if (epoch > 10) and ((epoch+1) % opt['save_epochs'] == 0):
            checkpoint = {
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
            
            torch.save(checkpoint, f'{ckp_dir}/{epoch+1}_{train_loss:.4f}.pth')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the denoiser")
    parser.add_argument("--config", type=str, default='./config/config2.yml')
    argspar = parser.parse_args()

    opt = utils.parse(argspar.config)
    # print(opt['train_dataset']['type'])
    utils.recursive_log(opt['log_file'], opt)
    train(opt)
