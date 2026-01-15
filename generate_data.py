import os
import numpy as np
from scapy.all import rdpcap, IP
from pathlib import Path
from tqdm import tqdm

# ================= 配置区域 =================
# ⚠️ 对应你截图里的文件夹名字，不要改
PCAP_ROOT = "pcap_unfiltered" 

# 输出文件名
OUTPUT_X = "X_raw.npy"
OUTPUT_Y = "y_raw.npy"

# 序列长度 (6000个包)
SEQUENCE_LENGTH = 6000
# ===========================================

def get_raw_sequence(pcap_file, max_len=6000):
    """读取 pcap 并返回统一长度的序列"""
    try:
        packets = rdpcap(str(pcap_file))
        if len(packets) == 0: return None
        
        # 提取包大小 (Packet Sizes)
        seq = [len(pkt) for pkt in packets if IP in pkt]
        if len(seq) == 0: return None

        # 截断或补零
        if len(seq) > max_len:
            seq = seq[:max_len]
        else:
            seq = seq + [0] * (max_len - len(seq))  
        return seq
    except Exception as e:
        return None

def process_directory(root_dir):
    X_list = []
    y_list = []
    
    # 标签映射: Podcast=1, Music=0
    content_type_map = {
        'podcast': 1,
        'rock': 0, 'rap': 0, 'edm': 0
    }

    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"❌ 错误: 找不到文件夹 '{root_dir}'")
        return None, None

    print(f"📂 正在读取文件夹: {root_path} ...")

    # 遍历所有子文件夹
    for genre_folder in root_path.iterdir():
        if not genre_folder.is_dir(): continue

        # 获取标签
        genre = genre_folder.name.lower()
        label = content_type_map.get(genre)
        if label is None: continue

        # 找 pcap 文件
        pcap_files = list(genre_folder.glob('*.pcap')) + list(genre_folder.glob('*.pcapng'))
        
        for pcap_file in tqdm(pcap_files, desc=f"   提取 {genre}"):
            seq = get_raw_sequence(pcap_file, SEQUENCE_LENGTH)
            if seq is not None:
                X_list.append(seq)
                y_list.append(label)

    return np.array(X_list), np.array(y_list)

if __name__ == "__main__":
    # 1. 运行提取
    X, y = process_directory(PCAP_ROOT)

    # 2. 保存结果
    if X is not None and len(X) > 0:
        # 增加一个维度 (N, 6000, 1) 以适配深度学习模型
        X = X[..., np.newaxis]
        
        np.save(OUTPUT_X, X)
        np.save(OUTPUT_Y, y)
        print("\n" + "="*30)
        print("✅ 成功！数据已准备好。")
        print(f"   X_raw.npy: {X.shape}")
        print(f"   y_raw.npy: {y.shape}")
        print("="*30)
    else:
        print("\n❌ 失败：没有提取到数据，请检查文件夹位置。")