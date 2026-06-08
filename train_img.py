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
from dataset import  AVA_Comment_LMDB_Dataset, AVA_Comment_Dataset_bert, AVA_Comment_Dataset_vit_bert
from util import EDMLoss, AverageMeter, set_up_seed, EDMLoss_r1, Balanced_l2_Loss
import option as option
import warnings
warnings.filterwarnings('ignore')


opt = option.init()
# //opt.save_path = ''
f = open(f'{opt.save_path}/log_img_test.txt', 'a')
opt.device = torch.device("cuda:{}".format(1))
opt.type = 'img'
# opt.batch_size = 256
opt.batch_size = 32
# opt.lr = 1e-4
opt.epochs = 15

# def adjust_learning_rate(params, optimizer, epoch):
#     """Sets the learning rate to the initial LR
#        decayed by 10 every 30 epochs"""
#     lr = params.init_lr * (0.1 ** (epoch // 30))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


def get_score(opt, y_pred):
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

def train_second_stage(opt, epoch, model, loader, optimizer, criterion, criterion1):
    model.train()
    emd_losses = AverageMeter()
    mse_losses = AverageMeter()
    prompt_losses = AverageMeter()
    prompt_acc_meter = AverageMeter()
    true_score = []
    pred_score = []
    loader = tqdm(loader)
    # loader = tqdm(loader, file=sys.stdout)
    for idx, (img, text, y) in enumerate(loader):

        img = img.to(opt.device)
        y = y.to(opt.device)
        # y_pred = model.train_second_stage_with_multi_features(img)

        # y_pred,similarity_scores = model.train_second_stage_with_anchored_prompts( 
        #    img, 
        #    return_similarity=True 
        # )
        y_pred,similarity_scores = model.forward_with_anchored_prompts_cross_attention( 
           img, 
           return_similarity=True 
        )
        # y_pred, similarity_scores = model.forward_with_anchored_prompts_token_cross_attention( 
        #     img, 
        #     return_similarity=True 
        #     )
        loss1 = criterion(p_target=y, p_estimate=y_pred)
        loss2 = criterion1(y, y_pred)
        loss_prompt, quality_label, quality_onehot = model.compute_prompt_loss(
            similarity_scores,
            y,
            target_is_distribution=True
        )
        loss = loss1 + loss2 * 10 + model.lambda_prompt * loss_prompt

        with torch.no_grad():
            prompt_pred = torch.argmax(similarity_scores, dim=1)
            prompt_acc = (prompt_pred == quality_label).float().mean()
            prompt_acc_meter.update(prompt_acc.item(), img.size(0))

            num_q = getattr(model, 'num_quality_prompts', 5)
            label_count = torch.bincount(
                quality_label.detach().cpu().long(),
                minlength=num_q
            )

            pred_count = torch.bincount(
                prompt_pred.detach().cpu().long(),
                minlength=num_q
            )

        pscore, pscore_np = get_score(opt, y_pred)
        tscore, tscore_np = get_score(opt, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        emd_losses.update(loss1.item(), img.size(0))
        mse_losses.update(loss2.item(), img.size(0))
        prompt_losses.update(loss_prompt.item(), img.size(0))
        loader.desc = "[train epoch {}] emd: {:.3f}, mse: {:.3f}, prompt: {:.3f}, pacc: {:.3f}".format(
            epoch,
            emd_losses.avg,
            mse_losses.avg,
            prompt_losses.avg,
            prompt_acc_meter.avg)

        if epoch == 0 and idx == 0:
            print("y_pred:", y_pred.shape)
            print("similarity_scores:", similarity_scores.shape)
            print("quality_label unique:", torch.unique(quality_label, return_counts=True))
            print("loss1:", loss1.item(), "loss2:", loss2.item(), "loss_prompt:", loss_prompt.item())

        if idx == 0:
            quality_names = ["terrible", "bad", "average", "good", "perfect"]
            print(f"\n[Prompt Debug][Epoch {epoch}]")
            # prompt_acc may be defined inside torch.no_grad()
            try:
                print(f"prompt_acc(batch): {prompt_acc.item():.4f}")
            except Exception:
                pass
            print("quality_label distribution:")
            for i, name in enumerate(quality_names):
                print(f"  {i}-{name}: {label_count[i].item()}")

            print("prompt_pred distribution:")
            for i, name in enumerate(quality_names):
                print(f"  {i}-{name}: {pred_count[i].item()}")

            print("similarity_scores mean per class:")
            sim_mean = similarity_scores.detach().mean(dim=0).cpu()
            for i, name in enumerate(quality_names):
                print(f"  {i}-{name}: {sim_mean[i].item():.4f}")

        pred_score += pscore_np.tolist()
        true_score += tscore_np.tolist()

    plcc_mean = pearsonr(pred_score, true_score)
    srcc_mean = spearmanr(pred_score, true_score)
    true_score = np.array(true_score)
    true_score_label = np.where(true_score <= 5.00, 0, 1)
    pred_score = np.array(pred_score)
    pred_score_label = np.where(pred_score <= 5.00, 0, 1)
    acc = accuracy_score(true_score_label, pred_score_label)
    print(f'lcc_mean: {plcc_mean[0]:.3f}, srcc_mean: {srcc_mean[0]:.3f}, acc: {acc:.3f}')

    return emd_losses.avg, prompt_losses.avg, prompt_acc_meter.avg

@torch.no_grad()
def test_second_stage(opt, epoch, model, loader, criterion):
    model.eval()
    emd_losses = AverageMeter()
    true_score = []
    pred_score = []
    loader = tqdm(loader)

    for idx, (img, text, y) in enumerate(loader):

        img = img.to(opt.device)
        y = y.to(opt.device)
        # y_pred = model.train_second_stage_with_multi_features(img)
        # y_pred = model.forward_with_anchored_prompts(img, return_similarity=False)
        y_pred = model.forward_with_anchored_prompts_cross_attention( img, return_similarity=False )
        # y_pred = model.forward_with_anchored_prompts_token_cross_attention( img, return_similarity=False )
        loss = criterion(p_target=y, p_estimate=y_pred)

        pscore, pscore_np = get_score(opt, y_pred)
        tscore, tscore_np = get_score(opt, y)

        emd_losses.update(loss.item(), img.size(0))
        loader.desc = "[test epoch {}] emd: {:.3f}".format(epoch, emd_losses.avg)

        pred_score += pscore_np.tolist()
        true_score += tscore_np.tolist()

    plcc_mean = pearsonr(pred_score, true_score)
    srcc_mean = spearmanr(pred_score, true_score)
    true_score = np.array(true_score)
    true_score_label = np.where(true_score <= 5.00, 0, 1)
    pred_score = np.array(pred_score)
    pred_score_label = np.where(pred_score <= 5.00, 0, 1)
    acc = accuracy_score(true_score_label, pred_score_label)
    print(f'lcc_mean: {plcc_mean[0]:.3f}, srcc_mean: {srcc_mean[0]:.3f}, acc: {acc:.3f}')

    return emd_losses.avg, plcc_mean[0], srcc_mean[0], acc

def full_queue(model, loader):
    loader = tqdm(loader)
    for idx, (img, text, y) in enumerate(loader):
        img = img.to(opt.device)
        y = y.to(opt.device)
        # tscore, tscore_np = get_score(opt, y)
        model.full_queue(img, text)

def start_train(opt):
    train_loader, test_loader = create_data_part(opt)
    type = opt.type
    model = Swin_Bert_vlmo_clip_mean_score_multi_features(device=opt.device, depth=2, model_type='base', type=type).to(opt.device)

    d = torch.load(
        'results/best_mean_plcc.pth',
        map_location='cpu')
    print(model.load_state_dict(d, strict=False))
    optimizer = torch.optim.AdamW(
        # model.parameters(),
        # model.get_anchored_prompt_param_groups(),
        model.get_anchored_prompt_cross_attention_param_groups(),
        # model.get_anchored_prompt_token_cross_attention_param_groups(),
        betas=(0.9, 0.99),
        weight_decay=1e-4
    )
    # print("prompt_context.requires_grad:", model.prompt_context.requires_grad)
    # print("prompt_fusion[0].weight.requires_grad:", model.prompt_fusion[0].weight.requires_grad)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=1e-6)
    # scheduler = cosine_scheduler(optimizer, opt.lr, 10000, len(train_loader) * opt.epochs)
    print(f"save_path: {opt.save_path}")
    criterion = EDMLoss().to(opt.device)
    criterion1 = torch.nn.MSELoss().to(opt.device)
    # criterion1 = Balanced_l2_Loss(opt.device).to(opt.device)

    best_acc, best_plcc, best_srcc, best_loss = 0, 0, 0, 100
    for e in range(opt.epochs):
        train_loss, train_prompt_loss, train_prompt_acc = train_second_stage(
            opt, epoch=e, model=model, loader=train_loader, optimizer=optimizer,
            criterion=criterion, criterion1=criterion1
        )

        torch.save(model.state_dict(), f'{opt.save_path}/latest.pth')

        test_loss, test_plcc, test_srcc, test_acc = test_second_stage(opt, epoch=e, model=model, loader=test_loader, criterion=criterion)
        scheduler.step()

        if best_acc < test_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f'{opt.save_path}/best_acc.pth')

        if best_plcc < test_plcc:
            best_plcc = test_plcc
            torch.save(model.state_dict(), f'{opt.save_path}/best_plcc.pth')

        if best_srcc < test_srcc:
            best_srcc = test_srcc
            torch.save(model.state_dict(), f'{opt.save_path}/best_srcc.pth')
       
        f.write(
            'epoch:%d,gate:%.4f, plcc:%.3f,srcc:%.3f,acc:%.3f, train_emd:%.4f, train_prompt:%.4f, train_prompt_acc:%.4f, test_loss:%.4f, lambda_prompt:%.4f\r\n'
            % (e, model.prompt_cross_gate.item(), test_plcc, test_srcc, test_acc, train_loss, train_prompt_loss, train_prompt_acc, test_loss, model.lambda_prompt)
        )
        f.write('\r\n')
        f.flush()

    f.close()

if __name__ == "__main__":
    #### train model
    set_up_seed()
    start_train(opt)
    #### test model
    # start_check_model(opt)
