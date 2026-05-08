import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
from paper7 import NoPaddingGLUPredictor, GatedLinearUnit, MultiScaleGLUBlock, AdaptiveSequenceProcessor

class CircRNAInferenceDataset(Dataset):
    """circRNA推理数据集"""
    def __init__(self, sequences, labels=None):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        item = {
            "sequence": torch.tensor(self.sequences[idx].astype(np.float32)),
            "seq_len": self.sequences[idx].shape[0]
        }
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx].astype(np.float32))
        return item

def collate_fn_variable_length(batch):
    """自定义批处理函数 - 不进行填充"""
    sequences = [item["sequence"] for item in batch]
    seq_lens = [item["seq_len"] for item in batch]
    
    result = {
        "sequences": sequences,
        "seq_lens": seq_lens
    }
    
    if "label" in batch[0]:
        labels = [item["label"] for item in batch]
        result["labels"] = labels
    
    return result

class ModelInference:
    """模型推理类"""
    def __init__(self, device):
        self.device = device
    
    def load_best_model(self, protein, model_class, input_dim=6, hidden_dims=[128, 256, 128], num_blocks=3):
        """加载最佳模型"""
        model_path = f'glu_results_hebing/{protein}/{protein}_best_overall_model.pth'
        
        if not os.path.exists(model_path):
            print(f"警告: 未找到 {protein} 的最佳模型文件: {model_path}")
            return None, None
        
        # 加载模型检查点
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 创建模型实例
        model = model_class(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_blocks=num_blocks
        ).to(self.device)
        
        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✅ 已加载 {protein} 的最佳模型 (第{checkpoint['fold']}折, epoch {checkpoint['epoch']})")
        print(f"   验证集性能 - F1: {checkpoint['val_f1']:.4f}, AUC: {checkpoint['val_auc']:.4f}")
        
        return model, checkpoint.get('best_threshold', 0.5)
    
    def predict_nucleotide_level(self, model, data_loader):
        """核苷酸级别预测 - 添加Sigmoid转换"""
        all_predictions = []
        all_true_labels = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc="预测中")):
                sequences = [seq.to(self.device) for seq in batch["sequences"]]
                seq_lens = batch["seq_lens"]
                
                # 获取预测结果 (logits)
                predictions, _ = model(sequences, seq_lens)
                
                # 收集预测结果 - 添加Sigmoid转换
                for pred in predictions:
                    # 将logits转换为概率 [0,1]
                    pred_prob = torch.sigmoid(pred).cpu().numpy()
                    all_predictions.append(pred_prob)
                
                # 如果有真实标签，也收集起来
                if "labels" in batch:
                    labels = batch["labels"]
                    for label in labels:
                        all_true_labels.append(label.cpu().numpy())
                
                # 清理缓存
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
        
        return all_predictions, all_true_labels

    def save_combined_results(self, protein, train_predictions, test_predictions, 
                            train_true_labels, test_true_labels, train_ids, test_ids, best_threshold):
        """保存合并的训练集和测试集结果"""
        output_dir = f'prediction_results_hebing/{protein}'
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 保存合并的预测概率 - 每行一个序列，每列一个核苷酸的预测概率
        combined_pred_file = f'{output_dir}/{protein}_combined_pred.csv'
        
        # 找到最大序列长度用于创建列名
        max_seq_length = 0
        all_predictions = train_predictions + test_predictions
        for pred_seq in all_predictions:
            max_seq_length = max(max_seq_length, len(pred_seq))
        
        # 创建列名：position_0, position_1, ..., position_n
        position_columns = [f'position_{i}' for i in range(max_seq_length)]
        columns = ['sequence_id', 'dataset_type'] + position_columns
        
        # 创建预测概率DataFrame
        pred_data = []
        
        # 添加训练集数据
        for i, pred_seq in enumerate(train_predictions):
            seq_id = train_ids[i] if i < len(train_ids) else f"train_{i}"
            row_data = {'sequence_id': seq_id, 'dataset_type': 'train'}
            # 为每个位置添加预测概率
            for pos, prob in enumerate(pred_seq):
                row_data[f'position_{pos}'] = f"{prob:.6f}"
            # 对于长度不足的序列，剩余位置填充空值
            for pos in range(len(pred_seq), max_seq_length):
                row_data[f'position_{pos}'] = ''
            pred_data.append(row_data)
        
        # 添加测试集数据
        for i, pred_seq in enumerate(test_predictions):
            seq_id = test_ids[i] if i < len(test_ids) else f"test_{i}"
            row_data = {'sequence_id': seq_id, 'dataset_type': 'test'}
            # 为每个位置添加预测概率
            for pos, prob in enumerate(pred_seq):
                row_data[f'position_{pos}'] = f"{prob:.6f}"
            # 对于长度不足的序列，剩余位置填充空值
            for pos in range(len(pred_seq), max_seq_length):
                row_data[f'position_{pos}'] = ''
            pred_data.append(row_data)
        
        # 保存预测概率CSV
        pred_df = pd.DataFrame(pred_data, columns=columns)
        pred_df.to_csv(combined_pred_file, index=False)
        print(f"✅ 合并预测概率已保存到: {combined_pred_file}")
        
        # 2. 保存合并的真实标签 - 每行一个序列，每列一个核苷酸的真实标签
        if train_true_labels and test_true_labels:
            combined_true_file = f'{output_dir}/{protein}_combined_true.csv'
            
            # 创建真实标签DataFrame
            true_data = []
            
            # 添加训练集数据
            for i, true_seq in enumerate(train_true_labels):
                seq_id = train_ids[i] if i < len(train_ids) else f"train_{i}"
                row_data = {'sequence_id': seq_id, 'dataset_type': 'train'}
                # 为每个位置添加真实标签
                for pos, label in enumerate(true_seq):
                    row_data[f'position_{pos}'] = str(int(label))
                # 对于长度不足的序列，剩余位置填充空值
                for pos in range(len(true_seq), max_seq_length):
                    row_data[f'position_{pos}'] = ''
                true_data.append(row_data)
            
            # 添加测试集数据
            for i, true_seq in enumerate(test_true_labels):
                seq_id = test_ids[i] if i < len(test_ids) else f"test_{i}"
                row_data = {'sequence_id': seq_id, 'dataset_type': 'test'}
                # 为每个位置添加真实标签
                for pos, label in enumerate(true_seq):
                    row_data[f'position_{pos}'] = str(int(label))
                # 对于长度不足的序列，剩余位置填充空值
                for pos in range(len(true_seq), max_seq_length):
                    row_data[f'position_{pos}'] = ''
                true_data.append(row_data)
            
            # 保存真实标签CSV
            true_df = pd.DataFrame(true_data, columns=columns)
            true_df.to_csv(combined_true_file, index=False)
            print(f"✅ 合并真实标签已保存到: {combined_true_file}")
        else:
            combined_true_file = None
        
        # 3. 保存合并的统计信息
        combined_stats_file = f'{output_dir}/{protein}_combined_stats.csv'
        
        stats_data = []
        
        # 添加训练集统计
        for i, pred_seq in enumerate(train_predictions):
            seq_id = train_ids[i] if i < len(train_ids) else f"train_{i}"
            seq_length = len(pred_seq)
            avg_pred = np.mean(pred_seq)
            max_pred = np.max(pred_seq)
            min_pred = np.min(pred_seq)
            positive_ratio = np.mean(np.array(pred_seq) > best_threshold)
            
            stats_data.append({
                'sequence_id': seq_id,
                'dataset_type': 'train',
                'sequence_length': seq_length,
                'avg_prediction': f"{avg_pred:.4f}",
                'max_prediction': f"{max_pred:.4f}",
                'min_prediction': f"{min_pred:.4f}",
                'positive_ratio_best_threshold': f"{positive_ratio:.4f}"
            })
        
        # 添加测试集统计
        for i, pred_seq in enumerate(test_predictions):
            seq_id = test_ids[i] if i < len(test_ids) else f"test_{i}"
            seq_length = len(pred_seq)
            avg_pred = np.mean(pred_seq)
            max_pred = np.max(pred_seq)
            min_pred = np.min(pred_seq)
            positive_ratio = np.mean(np.array(pred_seq) > best_threshold)
            
            stats_data.append({
                'sequence_id': seq_id,
                'dataset_type': 'test',
                'sequence_length': seq_length,
                'avg_prediction': f"{avg_pred:.4f}",
                'max_prediction': f"{max_pred:.4f}",
                'min_prediction': f"{min_pred:.4f}",
                'positive_ratio_best_threshold': f"{positive_ratio:.4f}"
            })
        
        # 保存统计信息CSV
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_csv(combined_stats_file, index=False)
        
        print(f"✅ 合并统计信息已保存到: {combined_stats_file}")
        print(f"   训练集序列数: {len(train_predictions)}")
        print(f"   测试集序列数: {len(test_predictions)}")
        print(f"   总序列数: {len(train_predictions) + len(test_predictions)}")
        print(f"   使用的阈值: {best_threshold:.4f}")
        print(f"   最大序列长度: {max_seq_length}")
        
        return combined_stats_file, combined_pred_file, combined_true_file

def read_id_list(fasta_file):
    """读取只包含ID的FASTA文件"""
    ids = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        ids.append(record.id)
    return ids

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 蛋白质列表
    protein_list = ['EIF4A3', 'EWSR1', 'FXR1', 'FXR2', 'IGF2BP2', 'IGF2BP3', 'MOV10']
    
    # 创建推理器
    inference = ModelInference(device)
    
    for protein in protein_list:
        print(f"\n{'='*60}")
        print(f"正在处理蛋白质: {protein}")
        print(f"{'='*60}")
        
        try:
            # 1. 加载最佳模型
            model, best_threshold = inference.load_best_model(protein, NoPaddingGLUPredictor)
            if model is None:
                continue
            
            print(f"最佳阈值: {best_threshold:.4f}")
            
            # 2. 加载数据
            print("加载数据...")
            
            # 文件路径
            # base_path = '/home/zhangying/jupyterlab/CircSite/nucleotide_level_dataset/nucleotide_level_dataset/'
            # train_file = f"{base_path}/{protein}/{protein}_train.fasta"
            # test_file = f"{base_path}/{protein}/{protein}_test.fasta"
            
            train_data_path = f'processed_data/{protein}/hebing_{protein}_train.npz'
            test_data_path = f'processed_data/{protein}/hebing_{protein}_test.npz'
            
            if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
                print(f"警告: 未找到 {protein} 的数据文件")
                continue
            
            # 加载标签数据
            train_data = np.load(train_data_path, allow_pickle=True)
            test_data = np.load(test_data_path, allow_pickle=True)
            
            # 提取标签
            train_labels = train_data['labels']
            test_labels = test_data['labels']
            
            # 读取真实的序列ID
            print("读取序列ID...")
            train_ids = train_data['sequence_names']
            test_ids = test_data['sequence_names']
            
            print(f"训练集ID数量: {len(train_ids)}, 标签数量: {len(train_labels)}")
            print(f"测试集ID数量: {len(test_ids)}, 标签数量: {len(test_labels)}")
            
            # 3. 创建数据加载器
            batch_size = 8
            
            # 训练集预测
            print("开始预测训练集...")
            train_sequences = list(train_data['sequences'])
            train_dataset = CircRNAInferenceDataset(train_sequences, train_labels)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                    shuffle=False, collate_fn=collate_fn_variable_length)
            
            train_predictions, train_true_labels = inference.predict_nucleotide_level(model, train_loader)
            
            # 测试集预测  
            print("开始预测测试集...")
            test_sequences = list(test_data['sequences'])
            test_dataset = CircRNAInferenceDataset(test_sequences, test_labels)
            test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                   shuffle=False, collate_fn=collate_fn_variable_length)
            
            test_predictions, test_true_labels = inference.predict_nucleotide_level(model, test_loader)
            
            # 4. 保存合并的结果
            print("保存合并的预测结果...")
            combined_stats_file, combined_pred_file, combined_true_file = inference.save_combined_results(
                protein, train_predictions, test_predictions, 
                train_true_labels, test_true_labels, train_ids, test_ids, best_threshold
            )
            
            # 5. 显示预测统计
            print(f"\n预测统计信息 (使用最佳阈值 {best_threshold:.4f}):")
            
            # 训练集统计
            train_all_probs = np.concatenate(train_predictions) if train_predictions else np.array([])
            train_positive_ratio = np.mean(train_all_probs > best_threshold) if len(train_all_probs) > 0 else 0
            
            print(f"训练集:")
            print(f"  - 序列数量: {len(train_predictions)}")
            print(f"  - 总核苷酸数: {sum(len(pred) for pred in train_predictions)}")
            print(f"  - 平均预测概率: {np.mean([np.mean(pred) for pred in train_predictions]):.4f}")
            print(f"  - 阳性核苷酸比例: {train_positive_ratio:.4f}")
            
            # 测试集统计
            test_all_probs = np.concatenate(test_predictions) if test_predictions else np.array([])
            test_positive_ratio = np.mean(test_all_probs > best_threshold) if len(test_all_probs) > 0 else 0
            
            print(f"测试集:")
            print(f"  - 序列数量: {len(test_predictions)}")
            print(f"  - 总核苷酸数: {sum(len(pred) for pred in test_predictions)}")
            print(f"  - 平均预测概率: {np.mean([np.mean(pred) for pred in test_predictions]):.4f}")
            print(f"  - 阳性核苷酸比例: {test_positive_ratio:.4f}")
            
            print(f"建议阈值: {best_threshold:.4f}")
            
            # 6. 保存阈值信息
            threshold_file = f'prediction_results_hebing/{protein}/{protein}_threshold_info.txt'
            with open(threshold_file, 'w') as f:
                f.write(f"Protein: {protein}\n")
                f.write(f"Best Threshold: {best_threshold:.4f}\n")
                f.write(f"Training Sequences: {len(train_predictions)}\n")
                f.write(f"Test Sequences: {len(test_predictions)}\n")
                f.write(f"Total Nucleotides: {sum(len(pred) for pred in train_predictions + test_predictions)}\n")
                f.write(f"Train Positive Ratio (threshold {best_threshold:.4f}): {train_positive_ratio:.4f}\n")
                f.write(f"Test Positive Ratio (threshold {best_threshold:.4f}): {test_positive_ratio:.4f}\n")
            
            # 清理内存
            del model
            torch.cuda.empty_cache()
            
            print(f"✅ {protein} 处理完成!")
            
        except Exception as e:
            print(f"❌ 处理 {protein} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("所有蛋白质预测完成！")
    print("预测结果保存在各自的 prediction_results_hebing/{protein}/ 目录下")
    print(f"{'='*60}")

# 确保导入原始模型中定义的类
if __name__ == "__main__":
    main()