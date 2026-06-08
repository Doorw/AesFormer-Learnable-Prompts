#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import requests
from bs4 import BeautifulSoup


DEFAULT_LABEL = "1 1 1 1 1 1 1 1 1 1"


def load_comments_from_json(json_dir):
    all_comments = defaultdict(list)
    source_files = defaultdict(set)

    json_dir = Path(json_dir)
    for json_path in sorted(json_dir.glob("*.json")):
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load JSON file {json_path}: {e}")

        for filename, comments in data.items():
            if not filename.lower().endswith(".jpg"):
                filename = f"{filename}.jpg"
            image_id = Path(filename).stem
            if not isinstance(comments, list):
                continue
            for c in comments:
                if not isinstance(c, str):
                    continue
                text = normalize_comment(c)
                if text:
                    all_comments[image_id].append(text)
            if image_id in all_comments:
                source_files[image_id].add(json_path.name)

    # deduplicate comments while preserving order
    deduped_comments = {}
    for image_id, comments in all_comments.items():
        seen = set()
        unique_comments = []
        for comment in comments:
            if comment not in seen:
                seen.add(comment)
                unique_comments.append(comment)
        deduped_comments[image_id] = unique_comments

    return deduped_comments, source_files


def normalize_comment(text):
    text = text.replace("\n", " ")
    text = re.sub(r"[\s\\]+", " ", text)
    text = text.strip()
    return text


def find_image_url(image_id, retries=3, pause=1.0):
    url = f"https://www.dpchallenge.com/image.php?IMAGE_ID={image_id}"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    }
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            html = response.text
            image_url = parse_image_url_from_html(html)
            if image_url:
                return image_url
        except Exception:
            pass
        if attempt < retries:
            time.sleep(pause)
    return None


def parse_image_url_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    candidates = []

    # Collect direct image page links and thumbnails
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "Copyrighted_Image_Reuse_Prohibited" in href and href.lower().endswith(".jpg"):
            candidates.append(normalize_image_url(href))

    for img in soup.find_all("img", src=True):
        src = img["src"]
        if "Copyrighted_Image_Reuse_Prohibited" in src and src.lower().endswith(".jpg"):
            candidates.append(normalize_image_url(src))

    if candidates:
        return select_best_dpc_image_url(candidates)

    # Final fallback: regex search
    m = re.search(r'https?://[^\"]+Copyrighted_Image_Reuse_Prohibited[^\"]+\.jpg', html)
    if m:
        return normalize_image_url(m.group(0))

    return None


def select_best_dpc_image_url(urls):
    def size_key(url):
        match = re.search(r"/(\d+)/Copyrighted_Image_Reuse_Prohibited_", url)
        if match:
            return int(match.group(1))
        return 0

    return max(urls, key=size_key)


def normalize_image_url(url):
    if url.startswith("//"):
        return "https:" + url
    if url.startswith("http:"):
        return url.replace("http:", "https:", 1)
    return url


def download_image(image_url, output_path, retries=3, pause=1.0):
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            with requests.get(image_url, headers=headers, stream=True, timeout=30) as r:
                r.raise_for_status()
                with output_path.open("wb") as f:
                    for chunk in r.iter_content(1024 * 32):
                        if chunk:
                            f.write(chunk)
            return True
        except Exception:
            if attempt < retries:
                time.sleep(pause)
    return False


def build_dataset(json_dir, images_dir, output_csv, skip_download=False, max_images=None, label_placeholder=DEFAULT_LABEL, comment_separator=" \n "):
    comments_map, source_files = load_comments_from_json(json_dir)
    images_dir = Path(images_dir)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    image_ids = sorted(comments_map.keys(), key=lambda x: int(x) if x.isdigit() else x)
    if max_images:
        image_ids = image_ids[-max_images:]

    for image_id in image_ids:
        combined_comment = comment_separator.join(comments_map[image_id])
        image_filename = f"{image_id}.jpg"
        image_path = images_dir / image_filename
        row = {
            "image_id": image_id,
            "image_filename": image_filename,
            "image_path": os.path.relpath(image_path, Path.cwd()),
            "comments": combined_comment,
            "attributes": ",".join(sorted(source_files[image_id])),
            "label": label_placeholder,
        }

        if not skip_download:
            if not image_path.exists() or image_path.stat().st_size == 0:
                image_url = find_image_url(image_id)
                if image_url:
                    ok = download_image(image_url, image_path)
                    if ok:
                        rows.append(row)  # 只在下载成功时才添加
                else:
                    print(f"WARNING: skipping {image_id} - no URL found")
                    continue  # 跳过这个图片
            # else:
            #     rows.append(row)  # 已存在则添加
        else:
            rows.append(row)  # skip-download 模式添加所有

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "image_filename", "image_path", "comments", "attributes", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Dataset saved to {output_csv}")
    print(f"Images saved to {images_dir} (download skipped={skip_download})")
    print(f"Total records: {len(rows)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build DPC-Captions dataset from JSON comments and DPChallenge images.")
    parser.add_argument("--json-dir", default="data/DPC-Captions", help="Directory containing DPC-Captions JSON files")
    parser.add_argument("--images-dir", default="data/DPC-Captions/images", help="Directory to save downloaded images")
    parser.add_argument("--output-csv", default="data/DPC-Captions/dpc_captions_dataset.csv", help="Output CSV file for the merged dataset")
    parser.add_argument("--skip-download", action="store_true", help="Only build CSV without downloading images")
    parser.add_argument("--max-images", type=int, default=200000, help="Maximum number of images to process")
    parser.add_argument("--label-placeholder", default=DEFAULT_LABEL, help="Placeholder label string for compatibility with project dataset classes")
    parser.add_argument("--comment-separator", default=" \n ", help="Separator used to join multiple comments for a single image")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset(
        json_dir=args.json_dir,
        images_dir=args.images_dir,
        output_csv=args.output_csv,
        skip_download=args.skip_download,
        max_images=args.max_images,
        label_placeholder=args.label_placeholder,
        comment_separator=args.comment_separator,
    )
