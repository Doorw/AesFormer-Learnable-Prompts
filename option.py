import os


def init():
    class Opt:
        pass

    opt = Opt()

    # 数据路径（请根据你本地实验路径修改）
    opt.path_to_save_csv = os.getenv('PATH_TO_SAVE_CSV', '/home/zzd111/projects/AesFormer/data/DPC-Captions')
    opt.path_to_images = os.getenv('PATH_TO_IMAGES', '/home/zzd111/projects/AesFormer/data2/AVA_comment/images')
    opt.path_to_data = os.getenv('PATH_TO_DATA', '/home/zzd111/projects/AesFormer/data/DPC-Captions')
    # 结果保存路径
    opt.save_path = os.getenv('SAVE_PATH', '/home/zzd111/projects/AesFormer/results')

    # DataLoader 并行数
    opt.num_workers = int(os.getenv('NUM_WORKERS', '4'))
    # opt.swin_ckpt = os.getenv(
    # 'SWIN_CKPT',
    # os.path.expanduser('~/AESFORMER/checkpoints/swin/swin_base_patch4_window7_224.pth')
    # )

    # 训练参数默认值
    opt.device = os.getenv('DEVICE', 'cuda:0')
    opt.batch_size = int(os.getenv('BATCH_SIZE', '16'))
    opt.lr = float(os.getenv('LR', '1e-5'))
    opt.epochs = int(os.getenv('EPOCHS', '15'))
    opt.type = os.getenv('MODEL_TYPE', 'both')  
    opt.use_lmdb = bool(os.getenv('USE_LMDB', 'True'))

    # 兼容旧代码写法，如果save_path不存在则创建
    if opt.save_path and not os.path.isdir(opt.save_path):
        os.makedirs(opt.save_path, exist_ok=True)

    return opt


if __name__ == '__main__':
    print('option init:', vars(init()))
