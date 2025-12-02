# scripts/download_clip_model.py
"""手动下载CLIP模型到本地"""

import os
import requests
from pathlib import Path
from tqdm import tqdm

def download_file(url, save_path):
    """下载单个文件"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 如果已存在，跳过
    if save_path.exists():
        print(f"✓ Already exists: {save_path.name}")
        return True
    
    print(f"Downloading: {save_path.name}")
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(save_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"✓ Downloaded: {save_path.name}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to download {save_path.name}: {e}")
        if save_path.exists():
            save_path.unlink()
        return False


def download_clip_model(save_dir="./pretrained_models/clip-vit-base-patch32"):
    """下载CLIP模型所有文件"""
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 使用镜像站点
    base_url = "https://hf-mirror.com/openai/clip-vit-base-patch32/resolve/main"
    
    # 需要下载的文件列表
    files = [
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "pytorch_model.bin",
        "special_tokens_map.json",
    ]
    
    print("="*60)
    print("Downloading CLIP Model Files")
    print("="*60)
    print(f"Save to: {save_dir}")
    print(f"Using mirror: {base_url}")
    print()
    
    success_count = 0
    for filename in files:
        url = f"{base_url}/{filename}"
        save_path = save_dir / filename
        
        if download_file(url, save_path):
            success_count += 1
    
    print()
    print("="*60)
    if success_count == len(files):
        print(f"✓ All files downloaded successfully!")
        print(f"  Model saved to: {save_dir}")
    else:
        print(f"⚠️ Downloaded {success_count}/{len(files)} files")
        print(f"  Some files may be missing")
    print("="*60)
    
    return success_count == len(files)


if __name__ == "__main__":
    # 下载模型
    success = download_clip_model()
    
    if success:
        print("\nYou can now use the model with:")
        print('  model = CLIPModel.from_pretrained("./pretrained_models/clip-vit-base-patch32")')
