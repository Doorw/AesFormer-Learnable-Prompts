#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def load_ava_label_map(ava_dir):
    ava_dir = Path(ava_dir)
    label_map = {}
    for csv_path in sorted(ava_dir.glob("*.csv")):
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            expected_scores = [f"score{i}" for i in range(2, 12)]
            for row in reader:
                image_id = Path(row["image_id"]).stem
                scores = []
                missing = False
                for score_col in expected_scores:
                    value = row.get(score_col, "")
                    if value is None or value == "":
                        missing = True
                        break
                    try:
                        scores.append(str(int(float(value))))
                    except ValueError:
                        missing = True
                        break
                if missing or len(scores) != 10:
                    continue
                label_map[image_id] = " ".join(scores)
    return label_map


def update_labels(input_csv, label_map, output_csv, skip_missing=True):
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with input_csv.open("r", encoding="utf-8", newline="") as f_in, \
            output_csv.open("w", encoding="utf-8", newline="") as f_out:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise RuntimeError(f"No header found in {input_csv}")
        if "label" not in fieldnames:
            raise RuntimeError("Input CSV must contain a 'label' column")

        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        missing_ids = []
        kept_count = 0
        removed_count = 0
        
        for row in reader:  
            image_id = str(row.get("image_id", "")).strip()
            if image_id in label_map:
                row["label"] = label_map[image_id]
                writer.writerow(row)
                kept_count += 1
            else:
                missing_ids.append(image_id)
                removed_count += 1
                if not skip_missing:
                    print(f"WARNING: no AVA label for image_id={image_id}, removing this row")

    print(f"Updated CSV written to {output_csv}")
    print(f"Kept: {kept_count} rows, Removed: {removed_count} rows")
    if missing_ids:
        print(f"Missing AVA labels for {removed_count} images. First 20: {missing_ids[:20]}")


def parse_args():
    parser = argparse.ArgumentParser(description="Update DPC-Captions dataset CSV with AVA label distributions.")
    parser.add_argument("--ava-dir", default="data/AVA_data", help="Directory containing AVA_data CSV files")
    parser.add_argument("--input-csv", default="data/DPC-Captions/dpc_captions_dataset.csv", help="Input DPC-Captions CSV file")
    parser.add_argument("--output-csv", default="data/DPC-Captions/dpc_captions_dataset_with_ava_labels.csv", help="Output CSV file with updated labels")
    parser.add_argument("--skip-missing", action="store_true", help="Skip warnings for missing AVA labels")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    label_map = load_ava_label_map(args.ava_dir)
    print(f"Loaded {len(label_map)} AVA label entries from {args.ava_dir}")
    update_labels(args.input_csv, label_map, args.output_csv, skip_missing=args.skip_missing)
