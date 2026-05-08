import numpy as np
import os

def multi_scale_gc_content(sequence, window_sizes=[21, 51, 101]):
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
            features[i, j] = gc_count / len(window_seq)
    
    return features

def sequences_to_multi_scale_features(sequences, window_sizes=[21, 51, 101]):
    """
    将DNA序列转换为多尺度GC含量、氢键能力和环大小特征
    特征维度: len(window_sizes) + 2
    
    参数:
    - sequences: DNA序列列表
    - window_sizes: 多尺度窗口大小列表，默认为[21, 51, 101]bp
    
    返回:
    - 特征列表，每个元素是 (seq_length, len(window_sizes)+2) 的数组
    """
    all_features = []
    
    # 基于生物学知识的特征编码
    hydrogen_bond_capacity = {'A': 0.67, 'C': 1.0, 'G': 1.0, 'T': 0.67}
    ring_size = {'A': 1.0, 'C': 0.0, 'G': 1.0, 'T': 0.0}
    
    n_gc_features = len(window_sizes)
    total_features = n_gc_features + 2  # 多尺度GC + 氢键能力 + 环大小
    
    print(f"  使用多尺度窗口: {window_sizes} bp")
    print(f"  总特征维度: {total_features} (GC多尺度:{n_gc_features} + 氢键能力:1 + 环大小:1)")
    
    for seq in sequences:
        seq_length = len(seq)
        features = np.zeros((seq_length, total_features), dtype=np.float32)
        
        # 1. 多尺度GC含量特征 (前n_gc_features维)
        gc_features = multi_scale_gc_content(seq, window_sizes)
        features[:, :n_gc_features] = gc_features
        
        # 2. 氢键能力和环大小特征 (后2维)
        for i, nucleotide in enumerate(seq):
            features[i, n_gc_features] = hydrogen_bond_capacity.get(nucleotide, 0.5)    # 氢键能力
            features[i, n_gc_features + 1] = ring_size.get(nucleotide, 0.5)             # 环大小
        
        all_features.append(features)
    
    return all_features

def analyze_multi_scale_pattern(sequence, window_sizes=[21, 51, 101]):
    """
    分析多尺度GC含量模式，用于验证多尺度环形窗口计算
    """
    seq_length = len(sequence)
    
    print(f"序列长度: {seq_length}")
    print(f"多尺度窗口: {window_sizes}")
    print("多尺度GC含量模式分析:")
    
    # 测试几个关键位置
    test_positions = [0, seq_length//4, seq_length//2, seq_length-1]
    
    for pos in test_positions:
        print(f"\n  位置 {pos}:")
        for window_size in window_sizes:
            half_window = window_size // 2
            
            # 构建环形窗口
            start = pos - half_window
            end = pos + half_window + 1
            
            window_seq = ""
            for p in range(start, end):
                circular_pos = p % seq_length
                if circular_pos < 0:
                    circular_pos += seq_length
                window_seq += sequence[circular_pos]
            
            gc_count = window_seq.count('G') + window_seq.count('C')
            gc_content = gc_count / len(window_seq)
            
            print(f"    窗口 {window_size}bp: GC含量 = {gc_content:.3f}")

def load_sequence_database(seq_file_path):
    """
    加载序列数据库，建立序列ID到序列的映射
    """
    seq_database = {}
    print(f"加载序列数据库: {seq_file_path}")
    
    with open(seq_file_path, 'r') as file:
        current_id = ''
        current_seq = ''
        
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                # 保存前一个序列
                if current_id and current_seq:
                    seq_database[current_id] = current_seq
                
                # 解析新序列的头信息
                header_parts = line[1:].split()
                current_id = header_parts[0]  # 取第一个部分作为序列ID
                current_seq = ''
            else:
                current_seq += line.upper()
        
        # 添加最后一个序列
        if current_id and current_seq:
            seq_database[current_id] = current_seq
    
    print(f"序列数据库加载完成，共 {len(seq_database)} 条序列")
    return seq_database

def parse_training_data_accumulated(file_path, seq_database):
    """
    解析训练/测试文件，累积同一个序列的所有结合区域
    返回去重的序列和累积的结合区域
    """
    # 使用字典来累积每个序列的所有结合区域
    sequence_regions_dict = {}
    sequence_data_dict = {}
    
    with open(file_path, 'r') as file:
        current_id = ''
        
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                # 解析头信息
                header_parts = line[1:].split()
                circrna_id = header_parts[0]  # 如 hsa_circ_0000002
                
                # 检查序列是否在数据库中
                if circrna_id not in seq_database:
                    print(f"警告: 未找到序列 {circrna_id} 在序列数据库中")
                    continue
                
                # 提取结合位点信息
                if len(header_parts) >= 4:
                    try:
                        start = int(header_parts[2])
                        end = int(header_parts[3])
                        
                        # 初始化或更新序列的结合区域列表
                        if circrna_id not in sequence_regions_dict:
                            sequence_regions_dict[circrna_id] = []
                            sequence_data_dict[circrna_id] = seq_database[circrna_id]
                        
                        # 添加新的结合区域
                        sequence_regions_dict[circrna_id].append((start, end))
                        
                    except ValueError:
                        print(f"警告: 无法解析结合区域信息: {line}")
    
    # 转换为列表格式
    sequences = []
    sequence_names = []
    binding_regions_list = []
    
    for seq_id, seq in sequence_data_dict.items():
        sequences.append(seq)
        sequence_names.append(seq_id)
        binding_regions_list.append(sequence_regions_dict[seq_id])
    
    print(f"解析完成: {len(sequences)} 个唯一序列，共 {sum(len(regions) for regions in binding_regions_list)} 个结合区域")
    
    # 打印序列长度统计
    lengths = [len(seq) for seq in sequences]
    print(f"序列长度范围: {min(lengths)}-{max(lengths)} bp, 中位数: {np.median(lengths):.0f} bp")
    
    return sequences, sequence_names, binding_regions_list

def create_accumulated_binding_labels(sequence_length, binding_regions):
    """
    为序列创建累积的结合位点标签
    处理多个重叠的结合区域，所有被任一实验标记的区域都设为1
    """
    labels = np.zeros(sequence_length, dtype=np.int8)
    
    if binding_regions:
        for region in binding_regions:
            start, end = region
            # 确保位置在有效范围内（从0开始索引）
            start_idx = max(0, start - 1)  # 从1-based转为0-based
            end_idx = min(sequence_length, end)  # 结束位置通常是包含的
            
            if start_idx < sequence_length and end_idx <= sequence_length and start_idx < end_idx:
                # 标记所有结合区域，允许重叠
                labels[start_idx:end_idx] = 1
    
    return labels

def analyze_binding_regions(sequence_names, binding_regions_list):
    """
    分析结合区域的统计信息
    """
    print("结合区域分析:")
    total_regions = sum(len(regions) for regions in binding_regions_list)
    total_sequences = len(sequence_names)
    
    regions_per_sequence = [len(regions) for regions in binding_regions_list]
    avg_regions = np.mean(regions_per_sequence) if regions_per_sequence else 0
    max_regions = max(regions_per_sequence) if regions_per_sequence else 0
    
    print(f"  总序列数: {total_sequences}")
    print(f"  总结合区域数: {total_regions}")
    print(f"  平均每个序列的结合区域数: {avg_regions:.2f}")
    print(f"  最大结合区域数: {max_regions}")
    print(f"  每个序列结合区域分布: {np.bincount(regions_per_sequence)}")

def process_protein_data_multi_scale(protein_name, output_dir='processed_data', window_sizes=[21, 51, 101]):
    """
    处理单个蛋白质的数据，使用多尺度GC含量、氢键能力和环大小特征编码
    支持累积结合区域和序列去重，适配circRNA环形结构
    """
    base_path = '/home/gyjsnnu/jupyterlab/CircSite/nucleotide_level_dataset/nucleotide_level_dataset/'
    train_file = f"{base_path}/{protein_name}/{protein_name}_train.fasta"
    test_file = f"{base_path}/{protein_name}/{protein_name}_test.fasta"
    seq_file = f"{base_path}/{protein_name}/{protein_name}_seq.fasta"
    
    print(f"\n处理 {protein_name} (多尺度窗口 {window_sizes}, 环形结构适配):")
    print(f"  训练文件: {train_file}")
    print(f"  测试文件: {test_file}")
    print(f"  序列文件: {seq_file}")
    
    # 检查文件是否存在
    if not os.path.exists(seq_file):
        print(f"错误: {protein_name} 的序列文件不存在")
        return None
    
    # 加载序列数据库
    seq_database = load_sequence_database(seq_file)
    
    # 创建输出目录
    protein_output_dir = os.path.join(output_dir, protein_name)
    os.makedirs(protein_output_dir, exist_ok=True)
    
    try:
        results = {}
        
        # 处理训练数据
        if os.path.exists(train_file) and os.path.getsize(train_file) > 0:
            print("  处理训练数据...")
            # 使用累积解析函数
            train_sequences, train_names, train_regions_list = parse_training_data_accumulated(train_file, seq_database)
            
            if train_sequences:
                # 分析结合区域
                analyze_binding_regions(train_names, train_regions_list)
                
                # 使用多尺度特征编码
                print(f"  生成多尺度GC含量、氢键能力和环大小特征...")
                train_features = sequences_to_multi_scale_features(train_sequences, window_sizes=window_sizes)
                
                # 验证多尺度计算
                if train_sequences:
                    print("  多尺度GC计算验证 (第一个序列):")
                    analyze_multi_scale_pattern(train_sequences[0], window_sizes)
                
                # 创建累积的结合标签
                train_labels = []
                for i, (seq, regions) in enumerate(zip(train_sequences, train_regions_list)):
                    labels = create_accumulated_binding_labels(len(seq), regions)
                    train_labels.append(labels)
                
                # 保存特征数据
                np.savez_compressed(
                    os.path.join(protein_output_dir, f'hebing_{protein_name}_train.npz'),
                    sequences=train_features,
                    labels=train_labels,
                    sequence_names=train_names,
                    binding_regions=train_regions_list,
                    original_sequences=train_sequences,
                    protein_name=protein_name,
                    feature_info=f"gc_content({len(window_sizes)} scales) + hydrogen_bond(1) + ring_size(1) = {len(window_sizes) + 2} dimensions",
                    window_sizes=window_sizes,
                    circular_structure=True
                )
                
                results['train'] = {
                    'samples': len(train_sequences),
                    'sequences': train_sequences,
                    'features': train_features,
                    'labels': train_labels,
                    'regions': train_regions_list
                }
                
                # 打印特征维度信息
                if train_features:
                    sample_feature = train_features[0]
                    print(f"  特征维度: {sample_feature.shape} (序列长度 × {len(window_sizes) + 2})")
                    print(f"  特征含义: [GC含量({len(window_sizes)}尺度), 氢键能力, 环大小]")
                    
            else:
                print("  训练数据解析后为空")
                results['train'] = {'samples': 0, 'sequences': [], 'features': [], 'labels': []}
        else:
            print("  训练文件不存在或为空")
            results['train'] = {'samples': 0, 'sequences': [], 'features': [], 'labels': []}
        
        # 处理测试数据
        if os.path.exists(test_file) and os.path.getsize(test_file) > 0:
            print("  处理测试数据...")
            test_sequences, test_names, test_regions_list = parse_training_data_accumulated(test_file, seq_database)
            
            if test_sequences:
                analyze_binding_regions(test_names, test_regions_list)
                test_features = sequences_to_multi_scale_features(test_sequences, window_sizes=window_sizes)
                
                test_labels = []
                for i, (seq, regions) in enumerate(zip(test_sequences, test_regions_list)):
                    labels = create_accumulated_binding_labels(len(seq), regions)
                    test_labels.append(labels)
                
                np.savez_compressed(
                    os.path.join(protein_output_dir, f'hebing_{protein_name}_test.npz'),
                    sequences=test_features,
                    labels=test_labels,
                    sequence_names=test_names,
                    binding_regions=test_regions_list,
                    original_sequences=test_sequences,
                    protein_name=protein_name,
                    feature_info=f"gc_content({len(window_sizes)} scales) + hydrogen_bond(1) + ring_size(1) = {len(window_sizes) + 2} dimensions",
                    window_sizes=window_sizes,
                    circular_structure=True
                )
                
                results['test'] = {
                    'samples': len(test_sequences),
                    'sequences': test_sequences,
                    'features': test_features,
                    'labels': test_labels,
                    'regions': test_regions_list
                }
        
        # 保存序列数据库
        seq_sequences = list(seq_database.values())
        seq_names = list(seq_database.keys())
        seq_features = sequences_to_multi_scale_features(seq_sequences, window_sizes=window_sizes)
        
        np.savez_compressed(
            os.path.join(protein_output_dir, f'hebing_{protein_name}_seq.npz'),
            sequences=seq_features,
            sequence_names=seq_names,
            original_sequences=seq_sequences,
            protein_name=protein_name,
            feature_info=f"gc_content({len(window_sizes)} scales) + hydrogen_bond(1) + ring_size(1) = {len(window_sizes) + 2} dimensions",
            window_sizes=window_sizes,
            circular_structure=True
        )
        
        results['seq'] = {
            'samples': len(seq_sequences),
            'sequences': seq_sequences,
            'features': seq_features,
            'labels': []
        }
        
        # 打印统计信息
        print(f"{protein_name} 多尺度特征统计:")
        for data_type in ['train', 'test', 'seq']:
            if data_type in results and results[data_type]['samples'] > 0:
                result = results[data_type]
                lengths = [len(seq) for seq in result['sequences']]
                feature_dims = result['features'][0].shape[1] if result['features'] else 0
                
                if data_type in ['train', 'test'] and result.get('labels'):
                    binding_bases = sum(np.sum(labels) for labels in result['labels'])
                    total_bases = sum(len(seq) for seq in result['sequences'])
                    binding_ratio = binding_bases / total_bases if total_bases > 0 else 0
                    
                    if result.get('regions'):
                        avg_regions = np.mean([len(regions) for regions in result['regions']])
                        print(f"  {data_type}: {result['samples']}样本, 长度{min(lengths)}-{max(lengths)}bp, 特征{feature_dims}维, 结合比例: {binding_ratio:.4f}, 平均结合区域: {avg_regions:.2f}")
        
        return {
            'protein_name': protein_name,
            'train_samples': results['train']['samples'],
            'test_samples': results['test']['samples'] if 'test' in results else 0,
            'seq_samples': results['seq']['samples'],
            'feature_dimension': len(window_sizes) + 2,
            'window_sizes': window_sizes,
            'circular_adapted': True
        }
        
    except Exception as e:
        print(f"处理 {protein_name} 时出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_multiple_proteins_multi_scale(protein_list, output_dir='processed_data', window_sizes=[21, 51, 101]):
    """
    处理多个蛋白质的数据集，使用多尺度GC含量、氢键能力和环大小特征编码
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"开始处理 {len(protein_list)} 个蛋白质的多尺度特征数据集")
    print("=" * 60)
    print(f"多尺度窗口: {window_sizes} bp")
    print(f"特征维度: {len(window_sizes) + 2} (GC多尺度:{len(window_sizes)} + 氢键能力:1 + 环大小:1)")
    print("适配circRNA环形结构")
    print("=" * 60)
    
    statistics = []
    successful_proteins = []
    
    for protein_name in protein_list:
        result = process_protein_data_multi_scale(protein_name, output_dir, window_sizes)
        if result is not None:
            statistics.append(result)
            successful_proteins.append(protein_name)
    
    # 打印总体统计
    print("\n" + "=" * 60)
    print("多尺度特征处理完成总结:")
    print(f"成功处理的蛋白质: {len(successful_proteins)}/{len(protein_list)}")
    print(f"多尺度窗口: {window_sizes} bp")
    print(f"特征维度: {len(window_sizes) + 2} 维")
    print(f"特征含义: [GC含量({len(window_sizes)}尺度), 氢键能力, 环大小]")
    print(f"成功的蛋白质列表: {successful_proteins}")
    
    if statistics:
        print("\n详细统计:")
        for stat in statistics:
            print(f"{stat['protein_name']}: 训练{stat['train_samples']}样本, 测试{stat['test_samples']}样本, 序列{stat['seq_samples']}样本, 特征{stat['feature_dimension']}维")
    
    return successful_proteins, statistics

# 使用示例
# if __name__ == "__main__":
#     protein_list = ['EIF4A3','EWSR1','FXR1','FXR2','IGF2BP2','IGF2BP3','MOV10']
    
#     # 设置多尺度窗口大小 - 针对200-6000bp序列优化
#     MULTI_SCALE_WINDOWS = [21, 51, 101, 201]  # 小、中、大、超大四个尺度
    
#     # 处理所有蛋白质数据
#     successful_proteins, stats = process_multiple_proteins_multi_scale(
#         protein_list, 
#         window_sizes=MULTI_SCALE_WINDOWS
#     )
    
#     print(f"\n多尺度特征处理完成! 成功处理 {len(successful_proteins)}/{len(protein_list)} 个蛋白质")
#     print(f"多尺度窗口: {MULTI_SCALE_WINDOWS} bp")
#     print(f"最终特征维度: {len(MULTI_SCALE_WINDOWS) + 2} 维")
#     print("输出文件命名格式: hebing_{protein_name}_{data_type}.npz")

def get_all_proteins_from_path(base_path):
    """从base_path目录下自动获取所有蛋白质名称"""
    proteins = []
    if os.path.exists(base_path):
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path):
                # 检查是否包含必要的fasta文件
                seq_file = os.path.join(item_path, f"{item}_seq.fasta")
                if os.path.exists(seq_file):
                    proteins.append(item)
    return sorted(proteins)

if __name__ == "__main__":
    base_path = '/home/gyjsnnu/jupyterlab/CircSite/nucleotide_level_dataset/nucleotide_level_dataset/'
    
    # 已处理的蛋白质
    processed_proteins = ['EIF4A3', 'EWSR1', 'FXR1', 'FXR2', 'IGF2BP2', 'IGF2BP3', 'MOV10']
    
    # 获取所有蛋白质
    all_proteins = get_all_proteins_from_path(base_path)
    print(f"发现 {len(all_proteins)} 个蛋白质文件夹")
    
    # 筛选出未处理的蛋白质
    remaining_proteins = [p for p in all_proteins if p not in processed_proteins]
    print(f"待处理蛋白质: {len(remaining_proteins)} 个")
    print(f"列表: {remaining_proteins}")
    
    # 设置多尺度窗口大小
    MULTI_SCALE_WINDOWS = [21, 51, 101, 201]
    
    # 处理剩余的蛋白质数据
    successful_proteins, stats = process_multiple_proteins_multi_scale(
        remaining_proteins, 
        window_sizes=MULTI_SCALE_WINDOWS
    )
    
    print(f"\n处理完成! 成功处理 {len(successful_proteins)}/{len(remaining_proteins)} 个蛋白质")