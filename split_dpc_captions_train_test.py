#!/usr/bin/env python3
import argparse
import csv
import random
from pathlib import Path


def split_csv(input_csv, train_csv, test_csv, val_csv, train_ratio=0.75, val_ratio=0.15, seed=42, shuffle=True):
    """
    将CSV文件划分为训练集、验证集和测试集
    
    Args:
        input_csv: 输入CSV文件路径
        train_csv: 训练集输出路径
        test_csv: 测试集输出路径
        val_csv: 验证集输出路径
        train_ratio: 训练集比例（默认0.75）
        val_ratio: 验证集比例（默认0.15）
        seed: 随机种子
        shuffle: 是否打乱数据
    """
    input_csv = Path(input_csv)
    train_csv = Path(train_csv)
    test_csv = Path(test_csv)
    val_csv = Path(val_csv)
    
    # 创建输出目录
    train_csv.parent.mkdir(parents=True, exist_ok=True)
    test_csv.parent.mkdir(parents=True, exist_ok=True)
    val_csv.parent.mkdir(parents=True, exist_ok=True)

    # 读取CSV
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = list(csv.DictReader(f))
        if not reader:
            raise RuntimeError(f"Input CSV {input_csv} is empty")
        fieldnames = reader[0].keys()

    # 打乱数据
    if shuffle:
        random.seed(seed)
        random.shuffle(reader)

    # 计算切分点
    total = len(reader)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    # 切分数据
    train_rows = reader[:train_end]
    val_rows = reader[train_end:val_end]
    test_rows = reader[val_end:]

    # 写入训练集
    with train_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(train_rows)

    # 写入验证集
    with val_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(val_rows)

    # 写入测试集
    with test_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(test_rows)

    # 打印统计信息
    print(f"Split {len(reader)} rows into:")
    print(f"  train: {len(train_rows)} rows ({len(train_rows)/total*100:.1f}%) -> {train_csv}")
    print(f"  val:   {len(val_rows)} rows ({len(val_rows)/total*100:.1f}%) -> {val_csv}")
    print(f"  test:  {len(test_rows)} rows ({len(test_rows)/total*100:.1f}%) -> {test_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="Split DPC-Captions CSV into train, validation and test sets.")
    parser.add_argument("--input-csv", default="data/DPC-Captions/dpc_captions_dataset_with_ava_labels.csv", 
                        help="Input CSV file")
    parser.add_argument("--train-csv", default="data/DPC-Captions/train.csv", 
                        help="Output train CSV file")
    parser.add_argument("--test-csv", default="data/DPC-Captions/test.csv", 
                        help="Output test CSV file")
    parser.add_argument("--val-csv", default="data/DPC-Captions/val.csv", 
                        help="Output validation CSV file")
    parser.add_argument("--train-ratio", type=float, default=0.75, 
                        help="Fraction of rows to use for train set (default: 0.75)")
    parser.add_argument("--val-ratio", type=float, default=0.15, 
                        help="Fraction of rows to use for validation set (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for shuffling")
    parser.add_argument("--no-shuffle", action="store_true", 
                        help="Do not shuffle before splitting")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # 验证比例之和是否为1
    total_ratio = args.train_ratio + args.val_ratio
    if total_ratio > 1.0:
        print(f"Warning: train_ratio + val_ratio = {total_ratio} > 1.0")
        print(f"Test ratio will be {1.0 - total_ratio}")
    
    split_csv(
        input_csv=args.input_csv,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        val_csv=args.val_csv,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )