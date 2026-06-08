import os
import pandas as pd
import lmdb
import pickle
import requests
from PIL import Image
from io import BytesIO
import time
import argparse
from tqdm import tqdm
import re
from bs4 import BeautifulSoup

def find_image_url(image_id, retries=3, pause=1.0):
    """从DPChallenge页面获取图片URL"""
    url = f"https://www.dpchallenge.com/image.php?IMAGE_ID={image_id}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            
            # 解析图片URL
            image_url = parse_image_url_from_html(response.text,image_id)
            if image_url:
                return image_url
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
        
        if attempt < retries - 1:
            time.sleep(pause)
    
    return None


def parse_image_url_from_html(html, target_image_id):
    """从HTML解析图片URL，优先选匹配目标ID的大图"""
    soup = BeautifulSoup(html, "html.parser")
    candidates = []

    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "Copyrighted_Image_Reuse_Prohibited" in href and href.lower().endswith(".jpg"):
            candidates.append(normalize_image_url(href))

    for img in soup.find_all("img", src=True):
        src = img["src"]
        if "Copyrighted_Image_Reuse_Prohibited" in src and src.lower().endswith(".jpg"):
            candidates.append(normalize_image_url(src))

    if candidates:
        return select_best_dpc_image_url(candidates, target_image_id)

    # 降级：正则匹配
    m = re.search(r'https?://[^\"]+Copyrighted_Image_Reuse_Prohibited[^\"]+\.jpg', html)
    if m:
        return normalize_image_url(m.group(0))

    return None


def select_best_dpc_image_url(urls, target_image_id):
    """从候选URL中选择属于目标图片的最大尺寸图片"""
    # 过滤：只保留包含目标ID的URL
    matched = [u for u in urls if f"_{target_image_id}.jpg" in u or f"/{target_image_id}.jpg" in u]
    
    if not matched:
        print(f"  Warning: No URL matches ID {target_image_id}, available: {[u[-40:] for u in urls]}")
        return None
    
    def size_key(url):
        match = re.search(r"/(\d+)/Copyrighted_Image_Reuse_Prohibited", url)
        return int(match.group(1)) if match else 0
    
    print(f"  Looking for ID: {target_image_id}")
    print(f"  Candidates: {[u[-50:] for u in urls]}")
    
    matched = [u for u in urls if f"_{target_image_id}.jpg" in u]
    print(f"  Matched: {[u[-50:] for u in matched] if matched else 'None'}")
    return max(matched, key=size_key)


def normalize_image_url(url):
    """规范化图片URL"""
    if url.startswith("//"):
        return "https:" + url
    if url.startswith("http:"):
        return url.replace("http:", "https:", 1)
    return url


def download_image(image_url, retries=3, pause=1.0):
    """下载图片"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    for attempt in range(retries):
        try:
            response = requests.get(image_url, headers=headers, timeout=30, stream=True)
            response.raise_for_status()
            return response.content
        except Exception as e:
            print(f"  Download attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(pause)
    
    return None


def resize_image(img_bytes, target_size=(224, 224)):
    """调整图片尺寸"""
    try:
        img = Image.open(BytesIO(img_bytes))
        img = img.convert('RGB')
        img = img.resize(target_size, Image.BICUBIC)
        
        # 转换回bytes
        output = BytesIO()
        img.save(output, format='JPEG', quality=85)
        return output.getvalue()
    except Exception as e:
        print(f"  Resize failed: {e}")
        return None


def convert_csv_to_lmdb(csv_path, lmdb_path, target_size=(224, 224), 
                        skip_existing=False, max_images=None, delay=0.5):
    """
    直接从CSV中的图片ID下载并转换为LMDB格式
    """
    # 读取CSV
    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total samples in CSV: {len(df)}")
    
    if max_images:
        df = df.head(max_images)    
        print(f"Limiting to {max_images} images")
    
    # 检查已有LMDB（用于断点续传）
    processed_ids = set()
    if skip_existing and os.path.exists(lmdb_path):
        print("Checking existing LMDB for already processed images...")
        try:
            # 使用 subdir=False 读取现有LMDB
            env_existing = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False)
            with env_existing.begin() as txn:
                cursor = txn.cursor()
                for key, _ in cursor:
                    processed_ids.add(key.decode())
            env_existing.close()
            print(f"Found {len(processed_ids)} already processed images")
        except Exception as e:
            print(f"Could not read existing LMDB: {e}")
    
    # 估算LMDB大小
    estimated_size_per_sample = 150 * 1024
    map_size = int(len(df) * estimated_size_per_sample * 2)
    
    print(f"Estimated LMDB size: {map_size / (1024**3):.2f} GB")
    
    # 创建LMDB环境 - 使用单文件模式
    env = lmdb.open(lmdb_path, map_size=map_size, 
                    subdir=False,  # 关键：使用单文件模式
                    readonly=False, 
                    lock=True)
    
    success_count = 0
    fail_count = 0
    skipped_count = 0
    
    with env.begin(write=True) as txn:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
            image_id = str(row['image_id'])
            
            if skip_existing and image_id in processed_ids:
                skipped_count += 1
                continue
            
            try:
                # 获取图片URL
                image_url = find_image_url(image_id)
                if not image_url:
                    print(f"\nWarning: Could not find URL for image_id={image_id}")
                    fail_count += 1
                    continue
                
                # 下载图片
                img_bytes = download_image(image_url)
                if not img_bytes:
                    print(f"\nWarning: Failed to download image_id={image_id}")
                    fail_count += 1
                    continue
                
                # 调整尺寸
                # resized_bytes = img_bytes
                resized_bytes = resize_image(img_bytes, target_size)
                if not resized_bytes:
                    print(f"\nWarning: Failed to resize image_id={image_id}")
                    fail_count += 1
                    continue
                
                # 获取评论和标签
                caption = row.get('comments', '')
                label = row.get('label', '')
                
                # 存储到LMDB
                data = pickle.dumps((resized_bytes, caption, label))
                txn.put(image_id.encode(), data)
                success_count += 1
                
                time.sleep(delay)
                
            except Exception as e:
                print(f"\nError processing {image_id}: {e}")
                fail_count += 1
                continue
            
            if (idx + 1) % 1000 == 0:
                print(f"\nProgress: {idx + 1}/{len(df)} | Success: {success_count} | Failed: {fail_count}")
    
    env.close()
    
    print(f"\n{'='*50}")
    print(f"Conversion completed!")
    print(f"Success: {success_count} samples")
    print(f"Failed: {fail_count} samples")
    print(f"Skipped (already in LMDB): {skipped_count} samples")
    print(f"LMDB saved to: {lmdb_path}")
    print(f"{'='*50}")


def verify_lmdb(lmdb_path, num_samples=5):
    """验证LMDB文件"""
    print(f"\nVerifying LMDB: {lmdb_path}")
    
    # 修正：添加 subdir=False，修复 stat() 调用
    env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False)
    
    with env.begin() as txn:
        # 修正：使用 txn.stat() 而不是 cursor.stat()
        total = txn.stat()['entries']
        print(f"Total samples in LMDB: {total}")
        
        if total == 0:
            print("Warning: LMDB is empty!")
            env.close()
            return
        
        print(f"\nFirst {min(num_samples, total)} samples:")
        cursor = txn.cursor()
        cursor.first()
        for i in range(min(num_samples, total)):
            key, value = cursor.item()
            data = pickle.loads(value)
            img_bytes, caption, label = data
            
            print(f"\nSample {i+1}:")
            print(f"  Image ID: {key.decode()}")
            print(f"  Image size: {len(img_bytes)} bytes ({len(img_bytes)/1024:.1f} KB)")
            print(f"  Caption preview: {caption[:100]}..." if len(caption) > 100 else f"  Caption: {caption}")
            print(f"  Label: {label[:50]}..." if len(label) > 50 else f"  Label: {label}")
            cursor.next()
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Download images from DPChallenge and convert to LMDB")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to CSV file (e.g., train.csv or test.csv)")
    parser.add_argument("--lmdb_path", type=str, required=True,
                        help="Output LMDB file path (e.g., train.lmdb)")
    parser.add_argument("--target_size", type=int, default=224,
                        help="Target image size (width=height)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip images already in LMDB (for resuming)")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Maximum number of images to process (for testing)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between requests in seconds")
    parser.add_argument("--verify", action="store_true",
                        help="Verify LMDB after conversion")
    
    args = parser.parse_args()
    
    # 转换
    convert_csv_to_lmdb(
        csv_path=args.csv_path,
        lmdb_path=args.lmdb_path,
        target_size=(args.target_size, args.target_size),
        skip_existing=args.skip_existing,
        max_images=args.max_images,
        delay=args.delay
    )
    
    # 验证
    if args.verify:
        verify_lmdb(args.lmdb_path)


if __name__ == "__main__":
    main()