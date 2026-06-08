import os
import re
import ast
import time
import argparse
from io import BytesIO

import requests
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
from tqdm import tqdm
from convert_to_lmdb import (
    find_image_url,
    download_image,
    parse_image_url_from_html,
    select_best_dpc_image_url,
    normalize_image_url,
)

import Code.option as option
from models.aesformer import Swin_Bert_vlmo_clip_mean_score_multi_features


def parse_label(label):
    """解析 label 字段，返回长度为 10 的归一化分布 numpy array"""
    if pd.isna(label):
        return None

    if isinstance(label, str):
        text = label.strip()
    elif isinstance(label, (list, tuple, np.ndarray)):
        arr = np.array(label, dtype=float)
        if arr.size != 10:
            return None
        arr = arr.astype(float)
        if arr.sum() == 0:
            return None
        return (arr / arr.sum()).astype(float)

    else:
        return None

    if not text:
        return None

    # 优先尝试 literal_eval
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, np.ndarray)):
            arr = np.array(parsed, dtype=float)
            if arr.size == 10:
                arr = arr.astype(float)
                if arr.sum() == 0:
                    return None
                return (arr / arr.sum()).astype(float)
    except Exception:
        pass

    # 备用文本提取
    numbers = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", text)
    if len(numbers) != 10:
        return None

    arr = np.array(numbers, dtype=float)
    if arr.sum() == 0:
        return None
    return (arr / arr.sum()).astype(float)


def ensure_distribution(tensor):
    """确保 tensor 是一个概率分布，必要时对其做 softmax"""
    if tensor.dim() == 2:
        dist = tensor
    elif tensor.dim() == 1:
        dist = tensor.unsqueeze(0)
    else:
        raise ValueError("Tensor must be 1D or 2D")

    with torch.no_grad():
        if torch.all(dist >= -1e-6):
            sums = dist.sum(dim=1, keepdim=True)
            if torch.allclose(sums, torch.ones_like(sums), atol=1e-3):
                return dist
        return torch.softmax(dist, dim=1)


def distribution_to_score(dist_np):
    scores = np.arange(1, 11, dtype=float)
    return float((dist_np * scores).sum())


def build_transform(input_size):
    return T.Compose([
        T.Resize((input_size, input_size), interpolation=Image.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def load_image_bytes(image_url):
    return download_image(image_url)


def save_image_bytes(image_bytes, save_path):
    try:
        with open(save_path, "wb") as f:
            f.write(image_bytes)
        return True
    except Exception as e:
        print(f"Warning: failed to save image bytes to {save_path}: {e}")
        return False


def save_processed_image(image, save_path):
    try:
        image.save(save_path, format="JPEG", quality=95)
        return True
    except Exception as e:
        print(f"Warning: failed to save processed image to {save_path}: {e}")
        return False


def load_model(checkpoint_path, opt):
    model = Swin_Bert_vlmo_clip_mean_score_multi_features(
        device=opt.device,
        depth=2,
        model_type="base",
        type=opt.type,
    ).to(opt.device)

    state_dict = torch.load(checkpoint_path, map_location=opt.device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def run_inference_one(model, transform, image, text, branch, device):
    image_tensor = transform(image).unsqueeze(0).to(device)
    text_input = text if isinstance(text, str) else str(text)
    with torch.no_grad():
        img_pred, text_pred = model.train_first_stage(image_tensor, text_input)

    if branch == "img":
        pred_dist = img_pred
    elif branch == "text":
        pred_dist = text_pred
    else:
        pred_dist = (img_pred + text_pred) / 2

    pred_dist = ensure_distribution(pred_dist)
    return pred_dist.squeeze(0).cpu().numpy()


def plot_single_result(image_path, gt_dist, pred_dist, mos, pred_score, save_path):
    image = Image.open(image_path).convert("RGB")
    fig = plt.figure(figsize=(6, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])

    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(image)
    ax_img.axis("off")

    ax_bar = fig.add_subplot(gs[1])
    xs = np.arange(1, 11)
    width = 0.25
    overlap = np.minimum(gt_dist, pred_dist)

    ax_bar.bar(xs - width, gt_dist, width=width, label="GT", color="#2a9d8f")
    ax_bar.bar(xs, pred_dist, width=width, label="Pred", color="#e76f51")
    ax_bar.bar(xs + width, overlap, width=width, label="Overlap", color="#264653")
    ax_bar.set_xticks(xs)
    ax_bar.set_xlabel("Score")
    ax_bar.set_ylabel("Probability")
    ax_bar.set_xlim(0.5, 10.5)
    ax_bar.set_ylim(0, max(gt_dist.max(), pred_dist.max(), overlap.max()) * 1.15)
    ax_bar.legend(loc="upper right", fontsize=9)
    ax_bar.set_title(f"MOS: {mos:.2f} / Pred: {pred_score:.2f}")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_grid_result(selected_samples, save_path_png, save_path_pdf, title="AVA"):
    n = len(selected_samples)
    if n == 0:
        return

    low_count = sum(1 for sample in selected_samples if sample["mos"] <= 5.0)
    high_count = n - low_count

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 6))
    if n == 1:
        axes = [axes]

    for idx, sample in enumerate(selected_samples):
        ax = axes[idx]
        panel_img = Image.open(sample["panel_path"]).convert("RGB")
        ax.imshow(panel_img)
        ax.axis("off")
        ax.set_title(f"ID: {sample['image_id']}\nMOS {sample['mos']:.2f} / Pred {sample['pred_score']:.2f}", fontsize=10)

    if low_count > 0:
        fig.text(0.05, 0.98, "Low Aesthetic Quality", fontsize=12, ha="left")
    if high_count > 0:
        fig.text(0.95, 0.98, "High Aesthetic Quality", fontsize=12, ha="right")
    if low_count > 0 and high_count > 0:
        boundary_x = (low_count / n) * 0.92 + 0.04
        fig.add_artist(Line2D([boundary_x, boundary_x], [0.05, 0.95], color="gray", linestyle="--", linewidth=1, transform=fig.transFigure))

    fig.suptitle(title, fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig.savefig(save_path_png, dpi=300)
    fig.savefig(save_path_pdf, dpi=300)
    plt.close(fig)


def choose_samples(results, num_low, num_high):
    low = [r for r in results if r["mos"] <= 5.0]
    high = [r for r in results if r["mos"] > 5.0]

    low_sorted = sorted(low, key=lambda x: x["abs_error"])[:num_low]
    high_sorted = sorted(high, key=lambda x: x["abs_error"])[:num_high]

    remaining = [r for r in results if r not in low_sorted and r not in high_sorted]
    remaining_sorted = sorted(remaining, key=lambda x: x["abs_error"]) if remaining else []

    if len(low_sorted) < num_low:
        extra = remaining_sorted[: num_low - len(low_sorted)]
        low_sorted.extend(extra)
        remaining_sorted = remaining_sorted[len(extra) :]

    if len(high_sorted) < num_high:
        extra = remaining_sorted[: num_high - len(high_sorted)]
        high_sorted.extend(extra)
        remaining_sorted = remaining_sorted[len(extra) :]

    return low_sorted, high_sorted


def make_dirs(output_dir):
    originals_dir = os.path.join(output_dir, "originals")
    processed_dir = os.path.join(output_dir, "processed")
    panels_dir = os.path.join(output_dir, "panels")
    os.makedirs(originals_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(panels_dir, exist_ok=True)
    return originals_dir, processed_dir, panels_dir


def main():
    parser = argparse.ArgumentParser(description="Visualize EMD predictions for AesFormer")
    parser.add_argument("--csv_path", required=True, help="输入 CSV 文件路径")
    parser.add_argument("--checkpoint", required=True, help="模型权重路径")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--max_samples", type=int, default=100, help="最多处理多少张图片")
    parser.add_argument("--num_low", type=int, default=2, help="低美学样本数量")
    parser.add_argument("--num_high", type=int, default=2, help="高美学样本数量")
    parser.add_argument("--input_size", type=int, default=224, help="模型输入尺寸")
    parser.add_argument("--device", default="cuda", help="运行设备")
    parser.add_argument("--branch", choices=["img", "text", "mean"], default="mean", help="预测分支")
    parser.add_argument("--delay", type=float, default=0.5, help="下载延迟秒数")
    args = parser.parse_args()

    opt = option.init()
    opt.device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    if opt.device.type == "cpu":
        print("CUDA unavailable or not selected, using CPU.")

    opt.type = opt.type if hasattr(opt, "type") else "both"

    originals_dir, processed_dir, panels_dir = make_dirs(args.output_dir)
    results_csv_path = os.path.join(args.output_dir, "visualization_results.csv")
    overview_png_path = os.path.join(args.output_dir, "ava_emd_visualization.png")
    overview_pdf_path = os.path.join(args.output_dir, "ava_emd_visualization.pdf")

    model = load_model(args.checkpoint, opt)
    transform = build_transform(args.input_size)

    df = pd.read_csv(args.csv_path)
    records = []
    print(f"Total rows in CSV: {len(df)}")

    for idx, row in tqdm(df.iterrows(), total=min(len(df), args.max_samples), desc="Processing samples"):
        if idx >= args.max_samples:
            break

        image_id = str(row.get("image_id", "")).strip()
        comments = str(row.get("comments", "")) if pd.notna(row.get("comments", "")) else ""
        label_value = row.get("label", None)

        if not image_id:
            print(f"Warning: missing image_id at row {idx}")
            continue

        gt_dist = parse_label(label_value)
        if gt_dist is None:
            print(f"Warning: failed to parse label for image_id {image_id}")
            continue

        image_url = find_image_url(image_id)
        if not image_url:
            print(f"Warning: failed to find image URL for {image_id}")
            continue

        image_bytes = download_image(image_url)
        if not image_bytes:
            print(f"Warning: failed to download image for {image_id} from {image_url}")
            continue

        original_path = os.path.join(originals_dir, f"{image_id}_original.jpg")
        if not save_image_bytes(image_bytes, original_path):
            continue

        try:
            original_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            print(f"Warning: failed to open downloaded image for {image_id}: {e}")
            continue

        processed_image = original_image.resize((args.input_size, args.input_size), Image.BICUBIC)
        processed_path = os.path.join(processed_dir, f"{image_id}_input{args.input_size}.jpg")
        if not save_processed_image(processed_image, processed_path):
            continue

        try:
            pred_dist = run_inference_one(model, transform, processed_image, comments, args.branch, opt.device)
        except Exception as e:
            print(f"Warning: inference failed for {image_id}: {e}")
            continue

        gt_score = distribution_to_score(gt_dist)
        pred_score = distribution_to_score(pred_dist)
        abs_error = abs(gt_score - pred_score)

        panel_path = os.path.join(panels_dir, f"{image_id}_panel.png")
        plot_single_result(original_path, gt_dist, pred_dist, gt_score, pred_score, panel_path)

        records.append({
            "image_id": image_id,
            "mos": gt_score,
            "pred_score": pred_score,
            "abs_error": abs_error,
            "branch": args.branch,
            "image_url": image_url,
            "original_path": original_path,
            "processed_path": processed_path,
            "panel_path": panel_path,
        })

        time.sleep(args.delay)

    if not records:
        print("No samples were successfully processed.")
        return

    df_results = pd.DataFrame(records)
    df_results.to_csv(results_csv_path, index=False)
    print(f"Saved results CSV to {results_csv_path}")

    low_selected, high_selected = choose_samples(records, args.num_low, args.num_high)
    selected = low_selected + high_selected
    if not selected:
        print("No selected samples for overview.")
        return

    plot_grid_result(selected, overview_png_path, overview_pdf_path, title="AVA")
    print(f"Saved overview PNG to {overview_png_path}")
    print(f"Saved overview PDF to {overview_pdf_path}")


if __name__ == "__main__":
    main()
