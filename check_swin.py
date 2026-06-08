import torch
import os
from models.swin import SwinTransformer

def main():
    model = SwinTransformer(
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        num_classes=10
    )

    ckpt_path = os.path.expanduser('~/projects/AesFormer/checkpoints/swin/swin_base_patch4_window7_224.pth')

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

    # 删除分类头
    # for k in ['head.weight', 'head.bias']:
    #     if k in state_dict:
    #         del state_dict[k]

    msg = model.load_state_dict(state_dict, strict=False)

    print("====== LOAD RESULT ======")
    print("Missing:", msg.missing_keys)
    print("Unexpected:", msg.unexpected_keys)


if __name__ == '__main__':
    main()