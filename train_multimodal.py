# -*- coding: utf-8 -*-
"""
多模态图像美学评估训练脚本
使用 Swin Transformer 和 BERT 进行图像和文本的联合训练
支持图像、文本和平均分数的预测
"""

import os
import sys
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score
import torch.optim.lr_scheduler as lr_scheduler
from models.aesformer import Swin_Bert_vlmo_clip_mean_score_multi_features
from dataset import AVA_Comment_Dataset, AVA_Comment_Dataset_bert, AVA_Comment_Dataset_vit_bert,AVA_Comment_LMDB_Dataset
from util import EDMLoss, AverageMeter, set_up_seed, EDMLoss_r1
import option as option
import warnings
warnings.filterwarnings('ignore')


# 初始化选项
opt = option.init()
#opt.save_path = ''  # 保存路径，需设置
f = open(f'{opt.save_path}/log_test.txt', 'a')  # 日志文件
# opt.device = torch.device("cuda:{}".format(3))  # 使用 GPU 3
opt.type = 'both'  # 模型类型
opt.batch_size = 16  # 批大小
opt.lr = 1e-5  # 学习率
opt.epochs = 25  # 训练轮数

# def adjust_learning_rate(init_lr, optimizer, epoch):
#     """
#     调整学习率，每20个epoch衰减10倍
#     """
#     lr = init_lr * (0.1 ** (epoch // 20))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


def get_score(opt, y_pred):
    """
    根据预测分布计算加权分数
    使用权重从1到10对应评分1到10
    """
    w = torch.from_numpy(np.linspace(1, 10, 10))
    w = w.type(torch.FloatTensor)
    w = w.to(opt.device)

    w_batch = w.repeat(y_pred.size(0), 1)

    score = (y_pred * w_batch).sum(dim=1)
    score_np = score.data.cpu().numpy()
    return score, score_np


def create_data_part(opt):
    """创建训练和测试数据加载器，支持LMDB"""
    
    if opt.use_lmdb:  # 新增配置项
        # LMDB模式
        train_lmdb = os.path.join(opt.path_to_data, 'Train.lmdb')
        test_lmdb = os.path.join(opt.path_to_data, 'Test.lmdb')
         
        train_ds = AVA_Comment_LMDB_Dataset(
            train_lmdb, 
            if_train=True
        )
        test_ds = AVA_Comment_LMDB_Dataset(
            test_lmdb, 
            if_train=False
        )
    else:
        # CSV模式（原逻辑）
        train_csv = os.path.join(opt.path_to_save_csv, 'train.csv')
        test_csv = os.path.join(opt.path_to_save_csv, 'test.csv')
        
        train_ds = AVA_Comment_Dataset_bert(
            train_csv, 
            opt.path_to_images, 
            if_train=True
        )
        test_ds = AVA_Comment_Dataset_bert(
            test_csv, 
            opt.path_to_images, 
            if_train=False
        )
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=opt.batch_size, 
        num_workers=opt.num_workers, 
        shuffle=True, 
        drop_last=True
    )
    test_loader = DataLoader(
        test_ds, 
        batch_size=opt.batch_size, 
        num_workers=opt.num_workers, 
        shuffle=False
    )
    
    return train_loader, test_loader

def train_first_stage_mean_score(opt, epoch, model, loader, optimizer, criterion):
    """
    第一阶段训练函数，使用平均分数
    训练图像和文本分支，并计算损失
    返回训练损失和评估指标
    """
    model.train()
    emd_losses = AverageMeter()
    img_losses = AverageMeter()
    text_losses = AverageMeter()
    itc_losses = AverageMeter()
    true_score = []
    img_pred_score = []
    text_pred_score = []
    mean_pred_score = []
    loader = tqdm(loader)
    for idx, (img, text, y) in enumerate(loader):

        img = img.to(opt.device)
        y = y.to(opt.device)

        img_pred, text_pred, itc_loss = model.train_first_stage(img, text)
        mean_pred = (img_pred + text_pred) / 2
        loss1 = criterion(p_target=y, p_estimate=img_pred)
        loss2 = criterion(p_target=y, p_estimate=text_pred)
        loss = loss1 + loss2 + itc_loss * 0.5

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        img_losses.update(loss1.item(), img.size(0))
        text_losses.update(loss2.item(), img.size(0))
        itc_losses.update(itc_loss.item(), img.size(0))

        loader.desc = "[train epoch {}] img_loss: {:.3f} text_loss: {:.3f} itc: {:.3f}".format(
                epoch, img_losses.avg, text_losses.avg, itc_losses.avg)

        img_pscore, img_pscore_np = get_score(opt, img_pred)
        text_pscore, text_pscore_np = get_score(opt, text_pred)
        mean_pscore, mean_pscore_np = get_score(opt, mean_pred)
        tscore, tscore_np = get_score(opt, y)

        img_pred_score += img_pscore_np.tolist()
        text_pred_score += text_pscore_np.tolist()
        mean_pred_score += mean_pscore_np.tolist()
        true_score += tscore_np.tolist()

    # 计算评估指标
    img_plcc_mean = pearsonr(img_pred_score, true_score)
    img_srcc_mean = spearmanr(img_pred_score, true_score)
    text_plcc_mean = pearsonr(text_pred_score, true_score)
    text_srcc_mean = spearmanr(text_pred_score, true_score)
    mean_plcc_mean = pearsonr(mean_pred_score, true_score)
    mean_srcc_mean = spearmanr(mean_pred_score, true_score)

    true_score = np.array(true_score)
    true_score_label = np.where(true_score <= 5.00, 0, 1)
    img_pred_score = np.array(img_pred_score)
    img_pred_score_label = np.where(img_pred_score <= 5.00, 0, 1)
    img_acc = accuracy_score(true_score_label, img_pred_score_label)
    print(f'img: lcc_mean: {img_plcc_mean[0]:.3f}, srcc_mean: {img_srcc_mean[0]:.3f}, acc: {img_acc:.3f}')

    text_pred_score = np.array(text_pred_score)
    text_pred_score_label = np.where(text_pred_score <= 5.00, 0, 1)
    text_acc = accuracy_score(true_score_label, text_pred_score_label)
    print(f'text: lcc_mean: {text_plcc_mean[0]:.3f}, srcc_mean: {text_srcc_mean[0]:.3f}, acc: {text_acc:.3f}')

    mean_pred_score = np.array(mean_pred_score)
    mean_pred_score_label = np.where(mean_pred_score <= 5.00, 0, 1)
    mean_acc = accuracy_score(true_score_label, mean_pred_score_label)
    print(f'mean: lcc_mean: {mean_plcc_mean[0]:.3f}, srcc_mean: {mean_srcc_mean[0]:.3f}, acc: {mean_acc:.3f}')

    plcc = {'img': img_plcc_mean[0], 'text': text_plcc_mean[0], 'mean': mean_plcc_mean[0]}
    srcc = {'img': img_srcc_mean[0], 'text': text_srcc_mean[0], 'mean': mean_srcc_mean[0]}
    acc = {'img': img_acc, 'text': text_acc, 'mean': mean_acc}

    return img_losses.avg, text_losses.avg, plcc, srcc, acc


@torch.no_grad()
def validate(opt, epoch, model, loader, criterion):
    """
    验证函数，计算测试集上的指标
    不进行梯度更新
    """
    model.eval()
    img_losses = AverageMeter()
    text_losses = AverageMeter()
    true_score = []
    img_pred_score = []
    text_pred_score = []
    mean_pred_score = []
    loader = tqdm(loader)
    # loader = tqdm(loader, file=sys.stdout)
    for idx, (img, text, y) in enumerate(loader):
        img = img.to(opt.device)
        y = y.to(opt.device)

        img_pred, text_pred = model.train_first_stage(img, text)
        r = 0.5
        mean_pred = img_pred * (1-r) + text_pred * r
        loss1 = criterion(p_target=y, p_estimate=img_pred)
        loss2 = criterion(p_target=y, p_estimate=text_pred)
        img_losses.update(loss1.item(), img.size(0))
        text_losses.update(loss2.item(), img.size(0))

        loader.desc = "[test epoch {}] img_loss: {:.3f} text_loss: {:.3f}".format(epoch, img_losses.avg, text_losses.avg)

        img_pscore, img_pscore_np = get_score(opt, img_pred)
        text_pscore, text_pscore_np = get_score(opt, text_pred)
        mean_pscore, mean_pscore_np = get_score(opt, mean_pred)
        tscore, tscore_np = get_score(opt, y)

        img_pred_score += img_pscore_np.tolist()
        text_pred_score += text_pscore_np.tolist()
        mean_pred_score += mean_pscore_np.tolist()
        true_score += tscore_np.tolist()

    img_plcc_mean = pearsonr(img_pred_score, true_score)
    img_srcc_mean = spearmanr(img_pred_score, true_score)
    text_plcc_mean = pearsonr(text_pred_score, true_score)
    text_srcc_mean = spearmanr(text_pred_score, true_score)
    mean_plcc_mean = pearsonr(mean_pred_score, true_score)
    mean_srcc_mean = spearmanr(mean_pred_score, true_score)

    true_score = np.array(true_score)
    true_score_label = np.where(true_score <= 5.00, 0, 1)
    img_pred_score = np.array(img_pred_score)
    img_pred_score_label = np.where(img_pred_score <= 5.00, 0, 1)
    img_acc = accuracy_score(true_score_label, img_pred_score_label)
    print(f'img: lcc_mean: {img_plcc_mean[0]:.3f}, srcc_mean: {img_srcc_mean[0]:.3f}, acc: {img_acc:.3f}')

    text_pred_score = np.array(text_pred_score)
    text_pred_score_label = np.where(text_pred_score <= 5.00, 0, 1)
    text_acc = accuracy_score(true_score_label, text_pred_score_label)
    print(f'text: lcc_mean: {text_plcc_mean[0]:.3f}, srcc_mean: {text_srcc_mean[0]:.3f}, acc: {text_acc:.3f}')

    mean_pred_score = np.array(mean_pred_score)
    mean_pred_score_label = np.where(mean_pred_score <= 5.00, 0, 1)
    mean_acc = accuracy_score(true_score_label, mean_pred_score_label)
    print(f'mean: lcc_mean: {mean_plcc_mean[0]:.3f}, srcc_mean: {mean_srcc_mean[0]:.3f}, acc: {mean_acc:.3f}')

    plcc = {'img': img_plcc_mean[0], 'text': text_plcc_mean[0], 'mean': mean_plcc_mean[0]}
    srcc = {'img': img_srcc_mean[0], 'text': text_srcc_mean[0], 'mean': mean_srcc_mean[0]}
    acc = {'img': img_acc, 'text': text_acc, 'mean': mean_acc}

    return img_losses.avg, text_losses.avg, plcc, srcc, acc



def full_queue(model, loader):
    """
    填充模型的队列，用于某些模型的预处理
    """
    loader = tqdm(loader)
    for idx, (img, text, y) in enumerate(loader):
        img = img.to(opt.device)
        y = y.to(opt.device)
        # tscore, tscore_np = get_score(opt, y)
        model.full_queue(img, text)

def start_train(opt):
    """
    开始训练过程
    初始化模型、优化器、调度器，循环训练并保存最佳模型
    """
    train_loader, test_loader = create_data_part(opt)
    type = opt.type
    model = Swin_Bert_vlmo_clip_mean_score_multi_features(device=opt.device, depth=2, model_type='base', type=type).to(opt.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, betas=(0.9, 0.99))
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=1e-6)

    criterion = EDMLoss().to(opt.device)

    best_mean_acc, best_mean_plcc, best_mean_srcc, best_mean_loss = 0, 0, 0, 100
    for e in range(opt.epochs):
        img_loss, text_loss, plcc, srcc, acc = train_first_stage_mean_score(opt,
                            epoch=e, model=model, loader=train_loader, optimizer=optimizer, criterion=criterion)

        torch.save(model.state_dict(), f'{opt.save_path}/latest.pth')

        test_img_loss, test_text_loss, test_plcc, test_srcc, test_acc = validate(opt,
                            epoch=e, model=model, loader=test_loader, criterion=criterion)
        scheduler.step()

        if best_mean_acc < test_acc['mean']:
            best_mean_acc = test_acc['mean']
            torch.save(model.state_dict(), f'{opt.save_path}/best_mean_acc.pth')

        if best_mean_srcc < test_srcc['mean']:
            best_mean_srcc = test_srcc['mean']
            torch.save(model.state_dict(), f'{opt.save_path}/best_mean_srcc.pth')

        if best_mean_plcc < test_plcc['mean']:
            best_mean_plcc = test_plcc['mean']
            torch.save(model.state_dict(), f'{opt.save_path}/best_mean_plcc.pth')

        f.write(
            'epoch:%d,img: lcc:%.3f,srcc:%.3f,acc:%.3f, train_loss:%.4f, tlcc:%.3f,tsrcc:%.3f,tacc:%.3f, test_loss:%.4f\r\n'
            % (e, plcc['img'], srcc['img'], acc['img'], img_loss, test_plcc['img'], test_srcc['img'], test_acc['img'],
               test_img_loss))

        f.write(
            'epoch:%d,text: lcc:%.3f,srcc:%.3f,acc:%.3f, train_loss:%.4f, tlcc:%.3f,tsrcc:%.3f,tacc:%.3f, test_loss:%.4f\r\n'
            % (e, plcc['text'], srcc['text'], acc['text'], text_loss, test_plcc['text'], test_srcc['text'],
               test_acc['text'], test_text_loss))
        f.write(
            'epoch:%d,mean: lcc:%.3f,srcc:%.3f,acc:%.3f, tlcc:%.3f,tsrcc:%.3f,tacc:%.3f\r\n'
            % (e, plcc['mean'], srcc['mean'], acc['mean'], test_plcc['mean'], test_srcc['mean'], test_acc['mean']))
        f.write('\r\n')
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {e}, LR: {current_lr:.2e}")
        f.flush()

    f.close()

if __name__ == "__main__":
    #### 训练模型
    set_up_seed()
    start_train(opt)
    #### 测试模型
    # start_check_model(opt)
