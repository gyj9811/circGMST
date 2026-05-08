import numpy as np
import os
import random
from collections import defaultdict

def multi_scale_gc_content(sequence, window_sizes=[21, 51, 101, 201]):
    """使用多个窗口大小计算GC含量，捕捉不同尺度的序列特征（环形结构适配）"""
    seq_length = len(sequence)
    features = np.zeros((seq_length, len(window_sizes)), dtype=np.float32)
    
    for j, window_size in enumerate(window_sizes):
        half_window = window_size // 2
        
        for i in range(seq_length):
            # 环形窗口计算
            start = i - half_window
            end = i + half_window + 1
            
            # 构建环形窗口序列
            window_seq = ""
            for pos in range(start, end):
                circular_pos = pos % seq_length
                if circular_pos < 0:
                    circular_pos += seq_length
                window_seq += sequence[circular_pos]
            
            # 计算GC含量
            gc_count = window_seq.count('G') + window_seq.count('C')
            features[i, j] = gc_count / len(window_seq) if len(window_seq) > 0 else 0
    
    return features

def sequence_to_multi_scale_features(sequence, window_sizes=[21, 51, 101, 201]):
    """
    将单个DNA序列转换为多尺度特征
    特征维度: len(window_sizes) + 2
    """
    # 基于生物学知识的特征编码
    hydrogen_bond_capacity = {'A': 0.67, 'C': 1.0, 'G': 1.0, 'T': 0.67, 'U': 0.67}
    ring_size = {'A': 1.0, 'C': 0.0, 'G': 1.0, 'T': 0.0, 'U': 0.0}
    
    n_gc_features = len(window_sizes)
    total_features = n_gc_features + 2
    
    seq = str(sequence).upper()
    seq_length = len(seq)
    features = np.zeros((seq_length, total_features), dtype=np.float32)
    
    # 1. 多尺度GC含量特征
    gc_features = multi_scale_gc_content(seq, window_sizes)
    features[:, :n_gc_features] = gc_features
    
    # 2. 氢键能力和环大小特征
    for i, nucleotide in enumerate(seq):
        features[i, n_gc_features] = hydrogen_bond_capacity.get(nucleotide, 0.5)
        features[i, n_gc_features + 1] = ring_size.get(nucleotide, 0.5)
    
    return features

def load_fasta_pool(fasta_file):
    """
    从FASTA文件中加载所有circRNA序列池
    """
    circrna_pool = {}
    print(f"正在加载FASTA文件: {fasta_file}")
    
    try:
        with open(fasta_file, 'r') as f:
            lines = f.readlines()
        
        for i in range(0, len(lines), 2):
            if i + 1 < len(lines):
                header_line = lines[i].strip()
                seq_line = lines[i+1].strip()
                
                if header_line.startswith('>'):
                    header = header_line[1:]  # 去掉'>'
                    
                    # 提取circRNA ID (格式: hsa_circ_0000001|chr1:1080738-1080845-|None|None)
                    parts = header.split('|')
                    circ_id = parts[0]  # hsa_circ_0000001
                    
                    seq_length = len(seq_line)
                    if 200 <= seq_length <= 6000:
                        circrna_pool[circ_id] = {
                            'header': header,
                            'sequence': seq_line.upper(),
                            'length': seq_length
                        }
        
        print(f"  成功加载 {len(circrna_pool)} 个序列（长度在200-6000之间）")
        if circrna_pool:
            print(f"  序列长度范围: {min(v['length'] for v in circrna_pool.values())} - "
                  f"{max(v['length'] for v in circrna_pool.values())}")
        
    except Exception as e:
        print(f"加载FASTA文件时出错: {e}")
        return {}
    
    return circrna_pool

def extract_sequence_ids_from_merged(protein_name, base_dir='processed_data'):
    """
    从合并的npz文件中提取序列ID
    """
    merged_file = os.path.join(base_dir, protein_name, f'hebing_{protein_name}_merged.npz')
    
    if not os.path.exists(merged_file):
        print(f"错误: 合并文件不存在 {merged_file}")
        return set(), 0
    
    try:
        data = np.load(merged_file, allow_pickle=True)
        
        # 检查文件包含的字段
        print(f"  合并文件字段: {list(data.files)}")
        
        # 查找序列ID字段
        sequence_ids = set()
        total_sequences = 0
        
        if 'sequence_names' in data:
            seq_names = data['sequence_names']
            total_sequences = len(seq_names)
            print(f"  找到 {total_sequences} 个序列名称")
            
            for name in seq_names:
                # 序列ID可能是不同的格式，提取hsa_circ_前缀的部分
                if isinstance(name, str):
                    # 如果格式是 hsa_circ_0000001|chr1:1080738-1080845-|None|None
                    parts = name.split('|')
                    circ_id = parts[0]  # hsa_circ_0000001
                    sequence_ids.add(circ_id)
        
        elif 'original_sequences' in data:
            # 如果没有sequence_names，尝试从original_sequences的header中提取
            # 这里需要根据实际数据结构调整
            pass
        
        print(f"  提取了 {len(sequence_ids)} 个唯一的序列ID")
        return sequence_ids, total_sequences
        
    except Exception as e:
        print(f"从合并文件提取序列ID时出错: {e}")
        return set(), 0

def select_negative_sequences(circrna_pool, positive_ids, num_needed):
    """
    从序列池中选择负样本序列
    """
    print(f"  需要选择 {num_needed} 个负样本序列")
    
    # 过滤掉与正样本重复的序列
    available_sequences = {}
    for circ_id, circ_info in circrna_pool.items():
        if circ_id not in positive_ids:
            available_sequences[circ_id] = circ_info
    
    print(f"  去除正样本重复后可用序列数: {len(available_sequences)}")
    
    if len(available_sequences) < num_needed:
        print(f"错误: 可用序列不足 ({len(available_sequences)} < {num_needed})")
        return []
    
    # 随机选择指定数量的序列
    available_list = list(available_sequences.items())
    random.shuffle(available_list)
    
    selected_sequences = available_list[:num_needed]
    
    print(f"  已随机选择 {len(selected_sequences)} 个序列")
    
    # 显示前5个选择的序列信息
    for i, (circ_id, circ_info) in enumerate(selected_sequences[:5]):
        print(f"    {i+1}. {circ_id} (长度: {circ_info['length']})")
    if len(selected_sequences) > 5:
        print(f"    ... 还有 {len(selected_sequences)-5} 个序列")
    
    return selected_sequences

def encode_negative_sequences(selected_sequences, window_sizes=[21, 51, 101, 201]):
    """
    编码负样本序列
    """
    print("  开始编码负样本序列...")
    
    sequences_encoded = []
    sequence_names = []
    original_sequences = []
    nucleotide_labels = []
    sequence_labels = []
    
    for i, (circ_id, circ_info) in enumerate(selected_sequences):
        # 编码序列特征
        encoded_seq = sequence_to_multi_scale_features(circ_info['sequence'], window_sizes)
        sequences_encoded.append(encoded_seq)
        
        # 序列名称（使用完整的header）
        sequence_names.append(circ_info['header'])
        
        # 原始序列
        original_sequences.append(circ_info['sequence'])
        
        # 核苷酸级别标签（全0，负样本）
        seq_length = circ_info['length']
        nucleotide_labels.append(np.zeros(seq_length, dtype=np.float32))
        
        # 序列级别标签（0，负样本）
        sequence_labels.append(0)
        
        if (i + 1) % 50 == 0:
            print(f"    已编码 {i + 1}/{len(selected_sequences)} 个序列")
    
    print(f"  编码完成，特征维度: {sequences_encoded[0].shape[1]}")
    
    return {
        'sequences': sequences_encoded,
        'sequence_names': sequence_names,
        'original_sequences': original_sequences,
        'nucleotide_labels': nucleotide_labels,
        'sequence_labels': sequence_labels
    }

def create_negative_dataset(protein_name, base_dir='processed_data', 
                           fasta_file='human_hg19_circRNAs_putative_spliced_sequence.fa',
                           window_sizes=[21, 51, 101, 201]):
    """
    创建负样本数据集
    """
    print(f"\n处理蛋白质: {protein_name}")
    print("=" * 60)
    
    # 1. 从合并文件中提取正样本序列ID
    print("1. 提取正样本序列ID...")
    positive_ids, num_needed = extract_sequence_ids_from_merged(protein_name, base_dir)
    
    if num_needed == 0:
        print("错误: 没有找到序列数据")
        return None
    
    # 2. 加载FASTA序列池
    print("\n2. 加载FASTA序列池...")
    circrna_pool = load_fasta_pool(fasta_file)
    
    if len(circrna_pool) == 0:
        print("错误: FASTA序列池为空")
        return None
    
    # 3. 选择负样本序列
    print("\n3. 选择负样本序列...")
    selected_sequences = select_negative_sequences(circrna_pool, positive_ids, num_needed)
    
    if len(selected_sequences) != num_needed:
        print(f"错误: 无法选择足够的负样本序列 ({len(selected_sequences)}/{num_needed})")
        return None
    
    # 4. 编码负样本序列
    print("\n4. 编码负样本序列...")
    negative_data = encode_negative_sequences(selected_sequences, window_sizes)
    
    # 5. 加载合并文件的元数据
    print("\n5. 加载元数据...")
    merged_file = os.path.join(base_dir, protein_name, f'hebing_{protein_name}_merged.npz')
    try:
        merged_data = np.load(merged_file, allow_pickle=True)
        
        # 提取元数据字段（非序列数据）
        meta_fields = ['protein_name', 'feature_info', 'window_sizes', 
                      'circular_structure', 'data_type', 'description',
                      'total_samples', 'source_files', 'has_sequence_labels']
        
        for field in meta_fields:
            if field in merged_data:
                negative_data[field] = merged_data[field]
        
        # 添加负样本特定元数据
        negative_data['data_type'] = 'negative_samples'
        negative_data['description'] = f'Negative samples for {protein_name}, selected from FASTA pool'
        negative_data['positive_sequence_ids'] = list(positive_ids)
        negative_data['negative_sequence_count'] = num_needed
        negative_data['window_sizes_used'] = window_sizes
        negative_data['feature_dimension'] = len(window_sizes) + 2
        
        print(f"  添加了 {len(meta_fields)} 个元数据字段")
        
    except Exception as e:
        print(f"  警告: 加载元数据时出错: {e}")
        # 添加基本元数据
        negative_data['protein_name'] = protein_name
        negative_data['data_type'] = 'negative_samples'
        negative_data['description'] = f'Negative samples for {protein_name}'
        negative_data['window_sizes'] = window_sizes
    
    return negative_data

def save_negative_dataset(negative_data, protein_name, base_dir='processed_data'):
    """
    保存负样本数据集
    """
    output_file = os.path.join(base_dir, protein_name, f'hebing_{protein_name}_negative.npz')
    
    # 确保目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"\n6. 保存负样本数据...")
    print(f"   保存到: {output_file}")
    
    # 准备保存的数据
    save_dict = {}
    for key, value in negative_data.items():
        save_dict[key] = value
    
    try:
        np.savez_compressed(output_file, **save_dict)
        
        # 验证保存的文件
        loaded_data = np.load(output_file, allow_pickle=True)
        print(f"   验证: 成功保存 {len(loaded_data.files)} 个字段")
        
        # 显示关键字段信息
        print(f"\n   关键字段统计:")
        for field in ['sequences', 'sequence_names', 'nucleotide_labels', 'sequence_labels']:
            if field in loaded_data:
                data = loaded_data[field]
                print(f"     {field}: {len(data)} 个")
                if field == 'sequences' and len(data) > 0:
                    print(f"          特征维度: {data[0].shape[1]}维")
        
        # 检查序列标签
        if 'sequence_labels' in loaded_data:
            labels = loaded_data['sequence_labels']
            if all(label == 0 for label in labels):
                print(f"    ✓ 序列标签验证: 所有{len(labels)}个标签均为0")
            else:
                print(f"    ⚠ 序列标签验证: 有{sum(1 for label in labels if label != 0)}个标签不为0")
        
        print(f"   ✓ {protein_name} 负样本数据保存完成")
        return output_file
        
    except Exception as e:
        print(f"   ✗ 保存失败: {e}")
        return None

def main():
    """主函数"""
    # 蛋白质列表
    protein_list = ['EIF4A3', 'EWSR1', 'FXR1', 'FXR2', 
                   'IGF2BP2', 'IGF2BP3', 'MOV10']
    
    # FASTA文件路径
    fasta_file = 'human_hg19_circRNAs_putative_spliced_sequence.fa'
    
    # 检查FASTA文件是否存在
    if not os.path.exists(fasta_file):
        print(f"错误: 未找到FASTA文件 {fasta_file}")
        print("请确保FASTA文件在当前目录或提供正确路径")
        return
    
    print("创建负样本数据集")
    print("=" * 70)
    print(f"FASTA文件: {fasta_file}")
    print(f"蛋白质列表: {', '.join(protein_list)}")
    print("=" * 70)
    
    # 窗口大小设置
    window_sizes = [21, 51, 101,201]
    print(f"使用窗口大小: {window_sizes}")
    print(f"特征维度: {len(window_sizes) + 2} (GC多尺度:{len(window_sizes)} + 氢键能力:1 + 环大小:1)")
    
    results = {}
    
    for protein in protein_list:
        print(f"\n{'='*70}")
        print(f"处理: {protein}")
        
        # 创建负样本数据集
        negative_data = create_negative_dataset(
            protein, 
            fasta_file=fasta_file,
            window_sizes=window_sizes
        )
        
        if negative_data is not None:
            # 保存负样本数据集
            saved_file = save_negative_dataset(negative_data, protein)
            
            if saved_file:
                results[protein] = {
                    'success': True,
                    'samples': len(negative_data['sequences']),
                    'file': saved_file
                }
            else:
                results[protein] = {'success': False, 'reason': '保存失败'}
        else:
            results[protein] = {'success': False, 'reason': '创建失败'}
    
    # 打印总结
    print(f"\n{'='*70}")
    print("处理完成总结:")
    print("=" * 70)
    
    success_count = sum(1 for result in results.values() if result.get('success', False))
    print(f"成功处理的蛋白质: {success_count}/{len(protein_list)}")
    
    print("\n详细结果:")
    for protein, result in results.items():
        status = "✓ 成功" if result.get('success', False) else "✗ 失败"
        if result.get('success', False):
            print(f"  {protein}: {status} ({result['samples']}个负样本)")
            print(f"    保存到: {result['file']}")
        else:
            print(f"  {protein}: {status} ({result.get('reason', '未知原因')})")

if __name__ == "__main__":
    main()