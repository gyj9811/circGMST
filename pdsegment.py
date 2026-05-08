# import numpy as np
# import pandas as pd
# from Bio import SeqIO
# import os

# def process_sequences_fasta(protein_name):
#     """
#     处理FASTA格式的序列数据，进行片段划分并赋予标签
    
#     Parameters:
#     protein_name: RBP蛋白名称
#     """
    
#     # 文件路径
#     base_path = '/home/zhangying/jupyterlab/CircSite/nucleotide_level_dataset/nucleotide_level_dataset/'
#     train_file = f"{base_path}/{protein_name}/{protein_name}_train.fasta"
#     test_file = f"{base_path}/{protein_name}/{protein_name}_test.fasta"
#     seq_file = f"{base_path}/{protein_name}/{protein_name}_seq.fasta"
    
#     # 加载标签数据
#     train_data = np.load(f'processed_data/{protein_name}/new_{protein_name}_train.npz', allow_pickle=True)
#     test_data = np.load(f'processed_data/{protein_name}/new_{protein_name}_test.npz', allow_pickle=True)
    
#     # 提取标签
#     train_labels = train_data['labels']
#     test_labels = test_data['labels']
    
#     def read_seq_mapping(seq_file):
#         """读取序列文件，建立ID到序列的映射"""
#         seq_mapping = {}
#         for record in SeqIO.parse(seq_file, "fasta"):
#             seq_id = record.id
#             sequence = str(record.seq)
#             seq_mapping[seq_id] = sequence
#         return seq_mapping
    
#     def read_id_list(fasta_file):
#         """读取只包含ID的FASTA文件"""
#         ids = []
#         for record in SeqIO.parse(fasta_file, "fasta"):
#             ids.append(record.id)
#         return ids
    
#     # 建立序列映射
#     print(f"正在读取序列文件...")
#     seq_mapping = read_seq_mapping(seq_file)
#     print(f"共读取 {len(seq_mapping)} 条序列")
    
#     # 读取训练和测试ID列表
#     train_ids = read_id_list(train_file)
#     test_ids = read_id_list(test_file)
    
#     print(f"训练ID数: {len(train_ids)}")
#     print(f"测试ID数: {len(test_ids)}")
    
#     def create_segments(id_list, labels_array, seq_mapping, dataset_type):
#         """创建50nt片段"""
#         all_segments = []
#         segment_length = 20
#         stride = 5  # 步长
        
#         for idx, seq_id in enumerate(id_list):
#             if seq_id not in seq_mapping:
#                 print(f"警告: {dataset_type} 序列 {seq_id} 在序列文件中未找到，跳过")
#                 continue
            
#             sequence = seq_mapping[seq_id]
#             seq_len = len(sequence)
            
#             # 检查标签索引
#             if idx >= len(labels_array):
#                 print(f"警告: {dataset_type} 序列 {seq_id} 的标签不存在，跳过")
#                 continue
            
#             labels = labels_array[idx]
            
#             # 检查序列长度与标签长度是否匹配
#             if seq_len != len(labels):
#                 print(f"警告: {dataset_type} 序列 {seq_id} 长度不匹配 (序列: {seq_len}, 标签: {len(labels)})，跳过")
#                 continue
            
#             # 检查序列长度是否足够
#             if seq_len < segment_length:
#                 print(f"{dataset_type} 序列 {seq_id} 长度不足 {segment_length}，跳过")
#                 continue
            
#             # 划分片段
#             for i in range(0, seq_len - segment_length + 1, stride):
#                 segment_seq = sequence[i:i+segment_length]
#                 segment_labels = labels[i:i+segment_length]
                
#                 # 计算片段标签（正样本比例）
#                 positive_ratio = np.mean(segment_labels)
                
#                 # 存储片段信息
#                 segment_info = {
#                     'sequence_id': seq_id,
#                     'segment_start': i,
#                     'segment_end': i+segment_length-1,
#                     'segment_sequence': segment_seq,
#                     'nucleotide_labels': ','.join([str(int(x)) for x in segment_labels]),
#                     'segment_label': positive_ratio,
#                     'dataset_type': dataset_type
#                 }
#                 all_segments.append(segment_info)
        
#         return all_segments
    
#     # 创建训练和测试片段
#     print("正在创建训练片段...")
#     train_segments = create_segments(train_ids, train_labels, seq_mapping, 'train')
#     print("正在创建测试片段...")
#     test_segments = create_segments(test_ids, test_labels, seq_mapping, 'test')
    
#     # 合并所有片段
#     all_segments = train_segments + test_segments
    
#     # 转换为DataFrame
#     df = pd.DataFrame(all_segments)
    
#     # 创建输出目录
#     output_dir = f'processed_data/{protein_name}'
#     os.makedirs(output_dir, exist_ok=True)
    
#     # 保存结果
#     output_file = f'{output_dir}/{protein_name}_20nt_segments_stride5.csv'
#     df.to_csv(output_file, index=False)
    
#     # 统计信息
#     train_pos_ratio = df[df['dataset_type'] == 'train']['segment_label'].mean()
#     test_pos_ratio = df[df['dataset_type'] == 'test']['segment_label'].mean()
    
#     print(f"\n处理完成!")
#     print(f"共生成 {len(all_segments)} 个片段")
#     print(f"训练片段: {len(train_segments)}, 平均正样本比例: {train_pos_ratio:.4f}")
#     print(f"测试片段: {len(test_segments)}, 平均正样本比例: {test_pos_ratio:.4f}")
#     print(f"结果已保存到 {output_file}")
    
#     return df

# # 使用示例
# if __name__ == "__main__":
#     # 替换为您的蛋白名称
#     protein_list = ['EIF4A3','EWSR1','FXR1','FXR2','IGF2BP2','IGF2BP3','MOV10']  # 替换为您的RBP名称
#     for protein in protein_list:
#         print(f"正在处理蛋白质: {protein}")
#         df_segments = process_sequences_fasta(protein)

import numpy as np
import pandas as pd
from Bio import SeqIO
import os

def process_sequences_fasta(protein_name):
    """
    处理FASTA格式的序列数据，进行片段划分并赋予标签
    读取累积保存的npz文件
    
    Parameters:
    protein_name: RBP蛋白名称
    """
    
    # 文件路径
    base_path = '/home/zhangying/jupyterlab/CircSite/nucleotide_level_dataset/nucleotide_level_dataset/'
    train_file = f"{base_path}/{protein_name}/{protein_name}_train.fasta"
    test_file = f"{base_path}/{protein_name}/{protein_name}_test.fasta"
    seq_file = f"{base_path}/{protein_name}/{protein_name}_seq.fasta"
    
    # 加载累积保存的标签数据
    train_data_path = f'processed_data/{protein_name}/hebing_{protein_name}_train.npz'
    test_data_path = f'processed_data/{protein_name}/hebing_{protein_name}_test.npz'
    
    if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
        print(f"错误: 未找到 {protein_name} 的累积npz文件")
        return None
    
    print(f"正在加载累积的npz文件...")
    train_data = np.load(train_data_path, allow_pickle=True)
    test_data = np.load(test_data_path, allow_pickle=True)
    
    # 提取数据
    train_labels = train_data['labels']
    train_sequence_names = train_data['sequence_names']
    train_original_sequences = train_data['original_sequences']
    train_binding_regions = train_data['binding_regions']
    
    test_labels = test_data['labels']
    test_sequence_names = test_data['sequence_names']
    test_original_sequences = test_data['original_sequences']
    test_binding_regions = test_data['binding_regions']
    
    print(f"训练数据: {len(train_sequence_names)} 个序列")
    print(f"测试数据: {len(test_sequence_names)} 个序列")
    
    def create_segments(sequence_names, labels_array, original_sequences, binding_regions, dataset_type):
        """创建20nt片段"""
        all_segments = []
        segment_length = 20
        stride = 5  # 步长
        
        for idx, seq_id in enumerate(sequence_names):
            if idx >= len(original_sequences) or idx >= len(labels_array):
                print(f"警告: {dataset_type} 序列 {seq_id} 数据不完整，跳过")
                continue
            
            sequence = original_sequences[idx]
            labels = labels_array[idx]
            
            # 安全地获取结合区域
            regions = []
            if idx < len(binding_regions):
                regions_item = binding_regions[idx]
                # 检查regions_item的类型
                if isinstance(regions_item, (list, np.ndarray)) and len(regions_item) > 0:
                    regions = regions_item
                elif regions_item is not None:
                    regions = [regions_item]
            
            seq_len = len(sequence)
            
            # 检查序列长度与标签长度是否匹配
            if seq_len != len(labels):
                print(f"警告: {dataset_type} 序列 {seq_id} 长度不匹配 (序列: {seq_len}, 标签: {len(labels)})，跳过")
                continue
            
            # 检查序列长度是否足够
            if seq_len < segment_length:
                print(f"{dataset_type} 序列 {seq_id} 长度不足 {segment_length}，跳过")
                continue
            
            # 划分片段
            for i in range(0, seq_len - segment_length + 1, stride):
                segment_seq = sequence[i:i+segment_length]
                segment_labels = labels[i:i+segment_length]
                
                # 计算片段标签（正样本比例）
                positive_ratio = np.mean(segment_labels)
                
                # 检查片段是否包含结合位点
                has_binding_site = np.any(segment_labels)
                
                # 存储片段信息 - 直接包含hsa名称和数据集类型
                segment_info = {
                    'hsa_sequence_id': seq_id,  # 直接显示hsa序列名称
                    'dataset_type': dataset_type,  # 直接显示训练集/测试集
                    'segment_start': i,
                    'segment_end': i+segment_length-1,
                    'segment_sequence': segment_seq,
                    'nucleotide_labels': ','.join([str(int(x)) for x in segment_labels]),
                    'segment_label': positive_ratio,
                    'has_binding_site': int(has_binding_site),
                    'binding_site_count': int(np.sum(segment_labels)),
                    'total_binding_regions': len(regions),
                    'segment_length': segment_length
                }
                all_segments.append(segment_info)
        
        return all_segments
    
    # 创建训练和测试片段
    print("正在创建训练片段...")
    train_segments = create_segments(train_sequence_names, train_labels, train_original_sequences, train_binding_regions, 'train')
    print("正在创建测试片段...")
    test_segments = create_segments(test_sequence_names, test_labels, test_original_sequences, test_binding_regions, 'test')
    
    # 合并所有片段
    all_segments = train_segments + test_segments
    
    # 转换为DataFrame
    df = pd.DataFrame(all_segments)
    
    # 创建输出目录
    output_dir = f'processed_data_hebing/{protein_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存片段结果 - 直接包含hsa名称和数据集类型
    segments_file = f'{output_dir}/{protein_name}_20nt_segments_stride5.csv'
    df.to_csv(segments_file, index=False)
    
    # 统计信息
    if len(all_segments) > 0:
        train_df = df[df['dataset_type'] == 'train']
        test_df = df[df['dataset_type'] == 'test']
        
        train_pos_ratio = train_df['segment_label'].mean()
        test_pos_ratio = test_df['segment_label'].mean()
        
        train_binding_sites = train_df['has_binding_site'].mean()
        test_binding_sites = test_df['has_binding_site'].mean()
        
        # 统计唯一的hsa序列数量
        unique_train_seqs = train_df['hsa_sequence_id'].nunique()
        unique_test_seqs = test_df['hsa_sequence_id'].nunique()
        
        print(f"\n处理完成!")
        print(f"共生成 {len(all_segments)} 个片段")
        print(f"训练集:")
        print(f"  片段数量: {len(train_segments)}")
        print(f"  唯一序列: {unique_train_seqs} 个")
        print(f"  平均正样本比例: {train_pos_ratio:.4f}")
        print(f"  包含结合位点的片段比例: {train_binding_sites:.4f}")
        print(f"测试集:")
        print(f"  片段数量: {len(test_segments)}")
        print(f"  唯一序列: {unique_test_seqs} 个")
        print(f"  平均正样本比例: {test_pos_ratio:.4f}")
        print(f"  包含结合位点的片段比例: {test_binding_sites:.4f}")
        print(f"片段数据已保存到: {segments_file}")
        
        # 打印一些示例信息
        print(f"\n示例片段信息:")
        sample_segments = df.head(3)
        for _, segment in sample_segments.iterrows():
            print(f"  {segment['hsa_sequence_id']} ({segment['dataset_type']}): "
                  f"位置{segment['segment_start']}-{segment['segment_end']}, "
                  f"标签: {segment['segment_label']:.3f}")
        
    else:
        print("警告: 未生成任何片段数据")
    
    return df

def analyze_npz_content(protein_name):
    """分析npz文件内容"""
    train_data_path = f'processed_data/{protein_name}/hebing_{protein_name}_train.npz'
    test_data_path = f'processed_data/{protein_name}/hebing_{protein_name}_test.npz'
    
    print(f"\n分析 {protein_name} 的npz文件内容:")
    print("=" * 50)
    
    for file_path, data_type in [(train_data_path, '训练'), (test_data_path, '测试')]:
        if os.path.exists(file_path):
            with np.load(file_path, allow_pickle=True) as data:
                print(f"{data_type}数据:")
                print(f"  序列数量: {len(data['sequence_names'])}")
                print(f"  标签数量: {len(data['labels'])}")
                print(f"  原始序列数量: {len(data['original_sequences'])}")
                if 'binding_regions' in data:
                    regions = data['binding_regions']
                    # 安全地计算结合区域统计
                    total_regions = 0
                    valid_regions_count = 0
                    
                    for region_item in regions:
                        if region_item is not None:
                            if isinstance(region_item, (list, np.ndarray)):
                                total_regions += len(region_item)
                                if len(region_item) > 0:
                                    valid_regions_count += 1
                            else:
                                total_regions += 1
                                valid_regions_count += 1
                    
                    avg_regions = total_regions / len(regions) if len(regions) > 0 else 0
                    print(f"  结合区域: 总{total_regions}个, 平均{avg_regions:.2f}个/序列")
                print(f"  序列名称示例: {data['sequence_names'][:3]}")
        else:
            print(f"{data_type}文件不存在: {file_path}")

# 使用示例
if __name__ == "__main__":
    protein_list = ['EIF4A3','EWSR1','FXR1','FXR2','IGF2BP2','IGF2BP3','MOV10']
    
    for protein in protein_list:
        print(f"\n{'='*60}")
        print(f"正在处理蛋白质: {protein}")
        print(f"{'='*60}")
        
        try:
            # 首先分析npz文件内容
            analyze_npz_content(protein)
            
            # 处理序列数据
            df_segments = process_sequences_fasta(protein)
            
            if df_segments is not None:
                print(f"✅ {protein} 处理完成!")
            else:
                print(f"❌ {protein} 处理失败!")
                
        except Exception as e:
            print(f"❌ 处理 {protein} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n所有蛋白质处理完成!")