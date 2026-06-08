# -*- coding: utf-8 -*-
"""
基于 CLIP + MLP 的美学回归训练脚本
使用 LMDB 数据集读取图像，冻结 CLIP 参数，仅训练回归头
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor

from dataset import AVA_Comment_LMDB_Dataset


class CLIPAestheticRegressor(nn.Module):
    """CLIP 图像编码 + MLP 回归头"""

    def __init__(self):
        super(CLIPAestheticRegressor, self).__init__()
        clip_path = "/home/zzd111/projects/AesFormer/pretrained/clip-vit-base-patch32"
        self.clip = CLIPModel.from_pretrained(clip_path)
        # 冻结 CLIP 参数
        for param in self.clip.parameters():
            param.requires_grad = False

        self.regressor = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, pixel_values):
        # 仅提取图像特征，不参与 CLIP 参数更新
        with torch.no_grad():
            image_features = self.clip.get_image_features(pixel_values)
        image_features = F.normalize(image_features, p=2, dim=-1)
        score_norm = self.regressor(image_features).squeeze(-1)
        return score_norm


def collate_fn(batch):
    images, captions, labels = zip(*batch)
    labels = np.stack(labels, axis=0).astype(np.float32)
    labels = torch.from_numpy(labels)
    return list(images), list(captions), labels


def score_from_distribution(dist):
    """从 1-10 的概率分布计算平均分"""
    bins = torch.arange(1, 11, device=dist.device, dtype=torch.float32)
    return (dist * bins).sum(dim=1)


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    pearson = pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else float('nan')
    spearman = spearmanr(y_true, y_pred)[0] if len(y_true) > 1 else float('nan')
    return mae, rmse, pearson, spearman


def train_one_epoch(model, processor, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_pred = []
    all_true = []

    for images, _, labels in loader:
        pixel_inputs = processor(images=images, return_tensors='pt').pixel_values.to(device)
        labels = labels.to(device)

        score = score_from_distribution(labels)
        target = (score - 1.0) / 9.0

        preds = model(pixel_inputs)
        loss = criterion(preds, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        all_pred.extend((preds.detach().cpu() * 9.0 + 1.0).tolist())
        all_true.extend(score.detach().cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    mae, rmse, pearson, spearman = compute_metrics(all_true, all_pred)
    return avg_loss, mae, rmse, pearson, spearman


@torch.no_grad()
def validate(model, processor, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_pred = []
    all_true = []

    for images, _, labels in loader:
        pixel_inputs = processor(images=images, return_tensors='pt').pixel_values.to(device)
        labels = labels.to(device)

        score = score_from_distribution(labels)
        target = (score - 1.0) / 9.0

        preds = model(pixel_inputs)
        loss = criterion(preds, target)

        total_loss += loss.item() * labels.size(0)
        all_pred.extend((preds.cpu() * 9.0 + 1.0).tolist())
        all_true.extend(score.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    mae, rmse, pearson, spearman = compute_metrics(all_true, all_pred)
    return avg_loss, mae, rmse, pearson, spearman


def parse_args():
    parser = argparse.ArgumentParser(description='CLIP + MLP 美学回归训练')
    parser.add_argument('--train_lmdb', type=str, default='data/DPC-Captions/train.lmdb', help='训练 LMDB 路径')
    parser.add_argument('--test_lmdb', type=str, default='data/DPC-Captions/test.lmdb', help='验证 LMDB 路径')
    parser.add_argument('--clip_path', type=str, default='/home/zzd111/projects/AesFormer/pretrained/clip-vit-base-patch32', help='CLIP 模型路径')
    parser.add_argument('--batch_size', type=int, default=16, help='批大小')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--eta_min', type=float, default=1e-6, help='余弦退火最小学习率')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader 线程数')
    parser.add_argument('--patience', type=int, default=5, help='早停耐心')
    parser.add_argument('--save_dir', type=str, default='results', help='模型保存目录')
    parser.add_argument('--device', type=str, default='cuda:2' if torch.cuda.is_available() else 'cpu', help='运行设备')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, 'train_clip_mlp_log.txt')

    device = torch.device(args.device)
    clip_path = args.clip_path

    processor = CLIPProcessor.from_pretrained(clip_path)
    model = CLIPAestheticRegressor().to(device)

    train_dataset = AVA_Comment_LMDB_Dataset(args.train_lmdb, if_train=True, transform=lambda img: img)
    val_dataset = AVA_Comment_LMDB_Dataset(args.test_lmdb, if_train=False, transform=lambda img: img)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(model.regressor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0

    with open(log_path, 'w') as log_file:
        log_file.write('epoch,lr,train_loss,train_mae,train_rmse,train_pearson,train_spearman,val_loss,val_mae,val_rmse,val_pearson,val_spearman\n')

        for epoch in range(1, args.epochs + 1):
            start = time.time()
            train_loss, train_mae, train_rmse, train_pearson, train_spearman = train_one_epoch(
                model, processor, train_loader, optimizer, criterion, device
            )

            val_loss, val_mae, val_rmse, val_pearson, val_spearman = validate(
                model, processor, val_loader, criterion, device
            )

            elapsed = time.time() - start
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch:02d} | time {elapsed:.1f}s | lr {current_lr:.6e}')
            print(f'  train loss={train_loss:.4f} mae={train_mae:.4f} rmse={train_rmse:.4f} pearson={train_pearson:.4f} spearman={train_spearman:.4f}')
            print(f'  valid loss={val_loss:.4f} mae={val_mae:.4f} rmse={val_rmse:.4f} pearson={val_pearson:.4f} spearman={val_spearman:.4f}')

            log_file.write(
                f'{epoch},{current_lr:.6e},{train_loss:.4f},{train_mae:.4f},{train_rmse:.4f},{train_pearson:.4f},{train_spearman:.4f},'
                f'{val_loss:.4f},{val_mae:.4f},{val_rmse:.4f},{val_pearson:.4f},{val_spearman:.4f}\n'
            )
            log_file.flush()

            scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_path = os.path.join(args.save_dir, 'best_hg_model.pth')
                torch.save(model.state_dict(), save_path)
                print(f'  best model saved: {save_path}')
            else:
                patience_counter += 1
                print(f'  no improvement, patience {patience_counter}/{args.patience}')

            if patience_counter >= args.patience:
                print('Early stopping triggered.')
                break

    final_save_path = os.path.join(args.save_dir, 'final_hg_model.pth')
    torch.save(model.state_dict(), final_save_path)
    print(f'Final model saved: {final_save_path}')
    print('Training finished. Best valid loss: {:.4f}'.format(best_val_loss))


if __name__ == '__main__':
    main()
