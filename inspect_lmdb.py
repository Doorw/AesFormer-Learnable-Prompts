# inspect_lmdb.py
import lmdb
import pickle
from PIL import Image
from io import BytesIO
import argparse

def inspect_lmdb(lmdb_path, num_samples=5, show_image=True):
    """查看LMDB中的内容"""
    
    env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False)
    
    with env.begin() as txn:
        # 修正：使用 txn.stat() 而不是 cursor.stat()
        total = txn.stat()['entries']
        print(f"📊 Total samples in LMDB: {total}")
        print(f"📁 LMDB file: {lmdb_path}")
        print(f"{'='*60}\n")
        
        if total == 0:
            print("⚠️  LMDB is empty!")
            env.close()
            return
        
        # 遍历前几个样本
        cursor = txn.cursor()
        cursor.first()
        for i in range(min(num_samples, total)):
            key, value = cursor.item()
            data = pickle.loads(value)
            img_bytes, caption, label = data
            
            print(f"📸 Sample {i+1}:")
            print(f"   🆔 Image ID: {key.decode()}")
            print(f"   📏 Image size: {len(img_bytes)} bytes ({len(img_bytes)/1024:.1f} KB)")
            print(f"   📝 Caption: {caption[:150]}..." if len(caption) > 150 else f"   📝 Caption: {caption}")
            print(f"   🏷️  Label: {label}")
            # 计算总投票数
            try:
                total_votes = sum(map(int, label.split()))
                print(f"   📊 Total votes: {total_votes}")
            except:
                pass
            print()
            
            # 可选：显示图片信息
            if show_image:
                img = Image.open(BytesIO(img_bytes))
                print(f"   🖼️  Image mode: {img.mode}, size: {img.size}")
                img.show()
            cursor.next()
    
    env.close()


def get_random_samples(lmdb_path, num_samples=3):
    """随机抽取几个样本查看"""
    
    import random
    
    env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False)
    
    with env.begin() as txn:
        total = txn.stat()['entries']
        
        if total == 0:
            print("LMDB is empty!")
            env.close()
            return
        
        # 获取所有keys
        keys = [key for key, _ in txn.cursor()]
        
        # 随机选择
        selected = random.sample(keys, min(num_samples, len(keys)))
        
        for key in selected:
            value = txn.get(key)
            data = pickle.loads(value)
            img_bytes, caption, label = data
            
            print(f"\n🎲 Random Sample:")
            print(f"   ID: {key.decode()}")
            print(f"   Caption: {caption[:100]}...")
            print(f"   Label: {label}")
            print(f"   Image size: {len(img_bytes)/1024:.1f} KB")
    
    env.close()


def search_by_id(lmdb_path, image_id):
    """根据图片ID查找"""
    
    env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False)
    
    with env.begin() as txn:
        key = str(image_id).encode()
        value = txn.get(key)
        
        if value:
            data = pickle.loads(value)
            img_bytes, caption, label = data
            print(f"✅ Found image ID: {image_id}")
            print(f"   📝 Caption: {caption[:200]}...")
            print(f"   🏷️  Label: {label}")
            print(f"   📏 Image size: {len(img_bytes)/1024:.1f} KB")
        else:
            print(f"❌ Image ID {image_id} not found")
    
    env.close()


def list_all_ids(lmdb_path, max_ids=20):
    """列出所有图片ID"""
    
    env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False)
    
    with env.begin() as txn:
        total = txn.stat()['entries']
        print(f"Total: {total} images")
        print(f"\nFirst {min(max_ids, total)} image IDs:")
        
        cursor = txn.cursor()
        cursor.first()
        for i in range(min(max_ids, total)):
            key, _ = cursor.item()
            print(f"  {i+1}. {key.decode()}")
            cursor.next()
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect LMDB file")
    parser.add_argument("--lmdb_path", type=str, required=True, help="Path to LMDB file")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to show")
    parser.add_argument("--show_image", action="store_true", help="Show image info")
    parser.add_argument("--random", action="store_true", help="Show random samples")
    parser.add_argument("--search", type=str, help="Search by image ID")
    parser.add_argument("--list_ids", action="store_true", help="List all image IDs")
    
    args = parser.parse_args()
    
    if args.search:
        search_by_id(args.lmdb_path, args.search)
    elif args.random:
        get_random_samples(args.lmdb_path, args.num_samples)
    elif args.list_ids:
        list_all_ids(args.lmdb_path, args.num_samples)
    else:
        inspect_lmdb(args.lmdb_path, args.num_samples, args.show_image)