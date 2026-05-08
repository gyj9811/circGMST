import numpy as np
import pandas as pd
import os
from tqdm import tqdm

def calculate_segment_predictions(protein_name):
    """
    根据核苷酸级别的预测结果计算每个片段的预测概率
    
    Parameters:
    protein_name: RBP蛋白名称
    """
    
    # 文件路径
    segment_file = f'processed_data_hebing/{protein_name}/{protein_name}_20nt_segments_stride5.csv'
    combined_pred_file = f'prediction_results_hebing/{protein_name}/{protein_name}_combined_pred.csv'
    
    # 检查文件是否存在
    if not all(os.path.exists(f) for f in [segment_file, combined_pred_file]):
        print(f"错误: 必要的文件不存在")
        print(f"片段文件: {segment_file} - {'存在' if os.path.exists(segment_file) else '不存在'}")
        print(f"预测文件: {combined_pred_file} - {'存在' if os.path.exists(combined_pred_file) else '不存在'}")
        return None
    
    # 读取片段数据
    print("正在读取片段数据...")
    df_segments = pd.read_csv(segment_file)
    
    # 读取合并的预测结果
    print("正在读取合并预测结果...")
    df_predictions = pd.read_csv(combined_pred_file)
    
    # 建立序列ID到预测结果的映射
    def create_prediction_mapping(df_predictions):
        """创建序列ID到预测结果的映射"""
        pred_mapping = {}
        
        for idx, row in df_predictions.iterrows():
            seq_id = row['sequence_id']
            
            # 提取所有位置的预测概率
            predictions = []
            for col in df_predictions.columns:
                if col.startswith('position_'):
                    prob_str = row[col]
                    if pd.notna(prob_str) and prob_str != '':
                        try:
                            predictions.append(float(prob_str))
                        except ValueError:
                            # 如果转换失败，跳过该位置
                            continue
            
            pred_mapping[seq_id] = predictions
        
        return pred_mapping
    
    print("正在建立预测结果映射...")
    pred_mapping = create_prediction_mapping(df_predictions)
    
    # 计算每个片段的预测概率
    print("正在计算片段预测概率...")
    
    all_segments = []
    missing_predictions = 0
    
    for idx, row in tqdm(df_segments.iterrows(), total=len(df_segments)):
        seq_id = row['hsa_sequence_id']
        start_pos = row['segment_start']
        end_pos = row['segment_end']
        dataset_type = row['dataset_type']
        segment_seq = row['segment_sequence']
        
        if seq_id in pred_mapping:
            predictions = pred_mapping[seq_id]
            
            # 检查位置是否有效
            if end_pos < len(predictions):
                # 提取片段的预测概率
                segment_probs = predictions[start_pos:end_pos+1]
                
                # 计算片段的平均预测概率
                segment_pred_prob = np.mean(segment_probs)
                
                # 存储片段信息
                segment_info = {
                    'sequence_id': seq_id,
                    'segment_start': start_pos,
                    'segment_end': end_pos,
                    'segment_sequence': segment_seq,
                    'segment_length': len(segment_seq),
                    'nucleotide_predictions': ','.join([f"{x:.6f}" for x in segment_probs]),
                    'segment_pred_prob': segment_pred_prob,
                    'dataset_type': dataset_type
                }
                all_segments.append(segment_info)
            else:
                missing_predictions += 1
        else:
            missing_predictions += 1
    
    # 转换为DataFrame
    result_df = pd.DataFrame(all_segments)
    
    # 创建输出目录
    output_dir = f'prediction_results_hebing/{protein_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存结果
    output_file = f'{output_dir}/{protein_name}_20nt_segments_predictions.csv'
    result_df.to_csv(output_file, index=False)
    
    print(f"\n处理完成!")
    print(f"共生成 {len(all_segments)} 个片段的预测结果")
    print(f"缺失预测的片段数: {missing_predictions}")
    print(f"结果已保存到: {output_file}")
    
    # 显示简单的统计信息
    if len(all_segments) > 0:
        print(f"\n片段预测概率统计:")
        print(f"平均预测概率: {result_df['segment_pred_prob'].mean():.4f}")
        print(f"预测概率范围: [{result_df['segment_pred_prob'].min():.4f}, {result_df['segment_pred_prob'].max():.4f}]")
    
    return result_df

# 批量处理所有蛋白质
def process_all_proteins():
    """批量处理所有蛋白质"""
    protein_list = ['EIF4A3', 'EWSR1', 'FXR1', 'FXR2', 'IGF2BP2', 'IGF2BP3', 'MOV10']
    
    for protein in protein_list:
        print(f"\n{'='*60}")
        print(f"正在处理蛋白质: {protein}")
        print(f"{'='*60}")
        
        try:
            result_df = calculate_segment_predictions(protein)
            if result_df is not None:
                print(f"✅ {protein} 处理完成! 生成 {len(result_df)} 个片段")
            
        except Exception as e:
            print(f"❌ 处理 {protein} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n所有蛋白质处理完成!")

# 使用示例
if __name__ == "__main__":
    # 方法1: 处理单个蛋白质
    # protein_name = 'EIF4A3'
    # result_df = calculate_segment_predictions(protein_name)
    
    # 方法2: 批量处理所有蛋白质
    process_all_proteins()