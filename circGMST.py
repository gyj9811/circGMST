import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, roc_auc_score
import os
import math
from roc1 import *
from metrics1 import *
from AUPR1 import *

class GatedLinearUnit(nn.Module):
    """门控线性单元 (GLU)"""
    def __init__(self, input_dim, output_dim):
        super(GatedLinearUnit, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim * 2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim) or (seq_len, input_dim)
        output = self.linear(x)
        gate, value = torch.chunk(output, 2, dim=-1)
        gate = self.sigmoid(gate)
        return gate * value

class MultiScaleGLUBlock(nn.Module):
    """多尺度GLU块，用于捕捉不同范围的依赖关系"""
    def __init__(self, input_dim, hidden_dims=[64, 128], dilations=[1, 2, 4]):
        super(MultiScaleGLUBlock, self).__init__()
        
        self.dilations = dilations
        self.conv_blocks = nn.ModuleList()
        
        for dilation in dilations:
            block = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dims[0], 3, padding=dilation, dilation=dilation),
                nn.BatchNorm1d(hidden_dims[0]),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Conv1d(hidden_dims[0], hidden_dims[1], 3, padding=dilation, dilation=dilation),
                nn.BatchNorm1d(hidden_dims[1]),
                nn.GELU()
            )
            self.conv_blocks.append(block)
        
        # 门控融合
        self.fusion_gate = GatedLinearUnit(hidden_dims[1] * len(dilations), hidden_dims[1])
        self.output_proj = nn.Linear(hidden_dims[1], hidden_dims[1])
        self.layer_norm = nn.LayerNorm(hidden_dims[1])
        
    def forward(self, x, seq_len):
        # x: (1, input_dim, seq_len)
        scale_outputs = []
        
        for i, conv_block in enumerate(self.conv_blocks):
            # 卷积处理
            conv_out = conv_block(x)  # (1, hidden_dim, seq_len)
            # 裁剪到原始长度
            conv_out = conv_out[:, :, :seq_len]
            scale_outputs.append(conv_out)
        
        # 多尺度特征融合
        fused = torch.cat(scale_outputs, dim=1)  # (1, hidden_dim * num_scales, seq_len)
        fused = fused.transpose(1, 2)  # (1, seq_len, hidden_dim * num_scales)
        
        # 门控融合
        fused = self.fusion_gate(fused)  # (1, seq_len, hidden_dim)
        fused = self.output_proj(fused)
        fused = self.layer_norm(fused)
        
        return fused.squeeze(0)  # (seq_len, hidden_dim)

class AdaptiveSequenceProcessor(nn.Module):
    """自适应序列处理器 - 核心GLU架构"""
    def __init__(self, input_dim=6, hidden_dims=[128, 256, 128], num_blocks=3):
        super(AdaptiveSequenceProcessor, self).__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        self.blocks = nn.ModuleList()
        
        # 创建多个处理块
        for i in range(num_blocks):
            block = nn.ModuleDict({
                'glu1': GatedLinearUnit(hidden_dims[i], hidden_dims[i]),
                'glu2': GatedLinearUnit(hidden_dims[i], hidden_dims[i]),
                'multi_scale': MultiScaleGLUBlock(hidden_dims[i], 
                                                [hidden_dims[i]//2, hidden_dims[i]],
                                                dilations=[1, 2, 4]),
                'layer_norm1': nn.LayerNorm(hidden_dims[i]),
                'layer_norm2': nn.LayerNorm(hidden_dims[i]),
                'dropout': nn.Dropout(0.1)
            })
            self.blocks.append(block)
            
            # 维度变换
            if i < num_blocks - 1:
                dim_change = nn.Linear(hidden_dims[i], hidden_dims[i+1])
                self.blocks.append(dim_change)
        
        # 输出层
        self.output_glu = GatedLinearUnit(hidden_dims[-1], hidden_dims[-1]//2)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dims[-1]//2, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1) 
        )
        
        self.embedding_dim = hidden_dims[-1]
        
    def forward(self, sequence, seq_len):
        # sequence: (seq_len, input_dim)
        x = self.input_proj(sequence)  # (seq_len, hidden_dim)
        
        block_idx = 0
        while block_idx < len(self.blocks):
            if isinstance(self.blocks[block_idx], nn.ModuleDict):
                # GLU处理块
                block = self.blocks[block_idx]
                
                # 第一个GLU + 残差
                residual = x
                glu1_out = block['glu1'](x)
                x = block['layer_norm1'](glu1_out + residual)
                
                # 多尺度处理
                x_conv = x.unsqueeze(0).transpose(1, 2)  # (1, hidden_dim, seq_len)
                multi_scale_out = block['multi_scale'](x_conv, seq_len)  # (seq_len, hidden_dim)
                
                # 第二个GLU + 残差
                residual = x
                glu2_out = block['glu2'](multi_scale_out)
                combined = glu2_out + multi_scale_out
                x = block['layer_norm2'](combined + residual)
                x = block['dropout'](x)
                
            else:
                # 维度变换
                x = self.blocks[block_idx](x)
            
            block_idx += 1
        
        # 输出处理
        x = self.output_glu(x)
        predictions = self.predictor(x).squeeze(-1)  
        return predictions

class NoPaddingGLUPredictor(nn.Module):
    """无填充的GLU预测模型"""
    def __init__(self, input_dim=6, hidden_dims=[128, 256, 128], num_blocks=3):
        super(NoPaddingGLUPredictor, self).__init__()
        self.processor = AdaptiveSequenceProcessor(input_dim, hidden_dims, num_blocks)
        
    def forward(self, sequences, seq_lens):
        """处理变长序列列表"""
        all_predictions = []
        
        for seq, seq_len in zip(sequences, seq_lens):
            # 单独处理每个序列
            predictions = self.processor(seq, seq_len)
            all_predictions.append(predictions)
        
        return all_predictions, None  # 返回None作为attention_weights的占位符

class CircRNADataset(Dataset):
   
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            "sequence": torch.tensor(self.sequences[idx].astype(np.float32)),
            "label": torch.tensor(self.labels[idx].astype(np.float32)),
            "seq_len": self.sequences[idx].shape[0]
        }

def collate_fn_variable_length(batch):
    """自定义批处理函数 - 不进行填充"""
    sequences = [item["sequence"] for item in batch]
    labels = [item["label"] for item in batch]
    seq_lens = [item["seq_len"] for item in batch]
    
    return {
        "sequences": sequences,
        "labels": labels,
        "seq_lens": seq_lens
    }

class NoPaddingGLUTrainer:
    """无填充GLU模型的训练器"""
    def __init__(self, model, RBP, device, num_folds=5):
        self.model = model
        self.RBP = RBP
        self.device = device
        self.num_folds = num_folds
        os.makedirs(f'glu_results_hebing/{self.RBP}', exist_ok=True)
    
    def _calculate_nucleotide_f1(self, all_true_labels, all_pred_labels, threshold=0.5):
        """计算核苷酸级别的F1分数"""
        all_true_flat = []
        all_pred_flat = []
        
        for true_labels, pred_labels in zip(all_true_labels, all_pred_labels):
            if len(true_labels) > 0 and len(pred_labels) > 0:
                pred_binary = (pred_labels > threshold).astype(int)
                all_true_flat.extend(true_labels)
                all_pred_flat.extend(pred_binary)
        
        if len(all_true_flat) == 0 or len(np.unique(all_true_flat)) < 2:
            return 0.0
        
        return f1_score(all_true_flat, all_pred_flat, zero_division=0)
    
    def _calculate_auc(self, all_true_labels, all_pred_labels):
        """计算AUC分数"""
        all_true_flat = []
        all_pred_flat = []
        
        for true_labels, pred_labels in zip(all_true_labels, all_pred_labels):
            if len(true_labels) > 0 and len(pred_labels) > 0:
                all_true_flat.extend(true_labels)
                all_pred_flat.extend(pred_labels)
        
        if len(all_true_flat) == 0 or len(np.unique(all_true_flat)) < 2:
            return 0.5
        
        return roc_auc_score(all_true_flat, all_pred_flat)
    
    def _find_best_threshold(self, all_true_labels, all_pred_labels):
        """寻找最佳阈值"""
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in np.arange(0.1, 0.9, 0.01):
            f1 = self._calculate_nucleotide_f1(all_true_labels, all_pred_labels, threshold)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return best_f1, best_threshold
    
    def train_epoch(self, train_loader, optimizer, criterion):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        for batch in train_loader:
            sequences = [seq.to(self.device) for seq in batch["sequences"]]
            labels = [label.to(self.device) for label in batch["labels"]]
            seq_lens = batch["seq_lens"]
            
            optimizer.zero_grad()
            
            # 前向传播
            predictions, _ = self.model(sequences, seq_lens)
            
            # 计算损失
            loss = 0
            valid_count = 0
            for pred, label in zip(predictions, labels):
                if len(pred) > 0 and len(pred) == len(label):
                    loss += criterion(pred, label.float()) 
                    valid_count += 1
            
            if valid_count > 0:
                loss = loss / valid_count
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
            
            # 清理缓存
            if batch_count % 10 == 0:
                torch.cuda.empty_cache()
        
        return total_loss / batch_count if batch_count > 0 else 0
    
    def evaluate(self, data_loader):
        """评估模型 - 保持序列结构"""
        self.model.eval()
        all_true_labels = []
        all_pred_labels = []

        with torch.no_grad():
            for batch in data_loader:
                sequences = [seq.to(self.device) for seq in batch["sequences"]]
                labels = [label.to(self.device) for label in batch["labels"]]
                seq_lens = batch["seq_lens"]

                predictions, _ = self.model(sequences, seq_lens)

                for pred, label in zip(predictions, labels):
                    if len(pred) > 0 and len(pred) == len(label):
                        # 添加sigmoid转换，将logits转为概率
                        pred_prob = torch.sigmoid(pred).cpu().numpy()  # 范围[0,1]
                        all_pred_labels.append(pred_prob)
                        all_true_labels.append(label.cpu().numpy())

                torch.cuda.empty_cache()

        return all_true_labels, all_pred_labels
    
    def _save_results_with_format(self, all_true_labels, all_pred_labels, current_cv, best_f1, best_auc):
          # 1. 保存真实标签文件
        true_file = f'glu_results_hebing/{self.RBP}/{self.RBP}_true_fold_{current_cv}.csv'
        with open(true_file, 'w') as f:
            for true_seq in all_true_labels:
                true_str = ','.join([str(int(x)) for x in true_seq])
                f.write(true_str + '\n')
        
        # 2. 保存预测概率文件
        pred_file = f'glu_results_hebing/{self.RBP}/{self.RBP}_pred_fold_{current_cv}.csv'
        with open(pred_file, 'w') as f:
            for pred_seq in all_pred_labels:
                pred_str = ','.join([f"{x:.6f}" for x in pred_seq])
                f.write(pred_str + '\n')
        
        # 3. 保存合并结果文件
        combined_file = f'glu_results_hebing/{self.RBP}/{self.RBP}_result_{current_cv}.csv'
        with open(combined_file, 'w') as f:
            f.write('prediction,true_label\n')
            for true_seq, pred_seq in zip(all_true_labels, all_pred_labels):
                for pred, true in zip(pred_seq, true_seq):
                    f.write(f"{pred:.6f},{int(true)}\n")
        
        print(f"第{current_cv}折结果:")
        print(f"  最佳F1: {best_f1:.4f}, AUC: {best_auc:.4f}")
        print(f"  序列数量: {len(all_true_labels)}, 核苷酸总数: {sum(len(seq) for seq in all_true_labels)}")
    
    def run_cross_validation(self, all_sequences, all_labels, batch_size=8, epochs=50, lr=0.001):
        """运行五折交叉验证"""
        dataset = CircRNADataset(all_sequences, all_labels)

        kfold = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        fold_results = []
        best_fold_models = []  # 存储每折的最佳模型信息
        
            # 计算正样本权重
        total_nucleotides = sum(len(label) for label in all_labels)
        positive_nucleotides = sum(torch.sum(torch.tensor(label)).item() for label in all_labels)
        pos_weight = (total_nucleotides - positive_nucleotides) / positive_nucleotides

        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            print(f"\n=== 第 {fold + 1} 折交叉验证 ===")

            # 划分数据集
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=batch_size, 
                                    shuffle=True, collate_fn=collate_fn_variable_length)
            val_loader = DataLoader(val_subset, batch_size=batch_size, 
                                  shuffle=False, collate_fn=collate_fn_variable_length)

            # 计算当前折训练集的pos_weight
            train_labels = [all_labels[i] for i in train_idx]
            total_nucleotides = sum(len(label) for label in train_labels)
            positive_nucleotides = sum(np.sum(label) for label in train_labels)

            # 避免除零错误
            if positive_nucleotides == 0:
                pos_weight = torch.tensor([1.0]).to(self.device)  # 如果没有正样本，使用中性权重
            else:
                pos_weight = torch.tensor([(total_nucleotides - positive_nucleotides) / positive_nucleotides]).to(self.device)

            print(f"  训练集正样本权重: {pos_weight.item():.2f}")
            print(f"  正样本数: {positive_nucleotides}/{total_nucleotides} ({positive_nucleotides/total_nucleotides*100:.2f}%)")

            # 重新初始化模型
            self.model.apply(self._reset_weights)
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            # 训练循环
            best_val_f1 = 0
            best_val_auc = 0
            best_epoch = 0
            best_fold_model_state = None

            for epoch in range(epochs):
                train_loss = self.train_epoch(train_loader, optimizer, criterion)
                scheduler.step()

                if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                    val_true_labels, val_pred_labels = self.evaluate(val_loader)

                    if val_true_labels and val_pred_labels:
                        val_f1 = self._calculate_nucleotide_f1(val_true_labels, val_pred_labels)
                        best_f1, best_threshold = self._find_best_threshold(val_true_labels, val_pred_labels)
                        val_auc = self._calculate_auc(val_true_labels, val_pred_labels)

                        print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
                        print(f'  Val F1: {val_f1:.4f}, Best F1: {best_f1:.4f}, AUC: {val_auc:.4f}')

                        if best_f1 > best_val_f1:
                            best_val_f1 = best_f1
                            best_val_auc = val_auc
                            best_epoch = epoch + 1
                            best_fold_model_state = self.model.state_dict().copy()
                            self.best_threshold = best_threshold

                torch.cuda.empty_cache()

            print(f"最佳模型在epoch {best_epoch}, F1: {best_val_f1:.4f}, AUC: {best_val_auc:.4f}")

            # 存储这折的最佳模型信息
            if best_fold_model_state is not None:
                best_fold_models.append({
                    'fold': fold + 1,
                    'f1': best_val_f1,
                    'auc': best_val_auc,
                    'epoch': best_epoch,
                    'model_state_dict': best_fold_model_state,
                    'threshold': self.best_threshold
                })

            # 使用最佳模型进行最终预测
            if best_fold_model_state is not None:
                self.model.load_state_dict(best_fold_model_state)

            final_true_labels, final_pred_labels = self.evaluate(val_loader)
            if final_true_labels and final_pred_labels:
                self._save_results_with_format(final_true_labels, final_pred_labels, 
                                             fold + 1, best_val_f1, best_val_auc)

            fold_results.append((best_val_f1, best_val_auc))

            # 清理
            if hasattr(self, 'best_model_state'):
                del self.best_model_state

        # 从五折中选择最佳模型
        if best_fold_models:
            # 按F1分数排序，选择最高的
            best_fold_models.sort(key=lambda x: x['f1'], reverse=True)
            best_overall_model = best_fold_models[0]

            # 保存五折中最佳模型
            best_model_save_path = f'glu_results_hebing/{self.RBP}/{self.RBP}_best_overall_model.pth'
            torch.save({
                'fold': best_overall_model['fold'],
                'epoch': best_overall_model['epoch'],
                'model_state_dict': best_overall_model['model_state_dict'],
                'val_f1': best_overall_model['f1'],
                'val_auc': best_overall_model['auc'],
                'best_threshold': best_overall_model['threshold'],
                'all_fold_results': fold_results
            }, best_model_save_path)
            print(f"\n🎯 五折中最佳模型已保存到: {best_model_save_path}")
            print(f"   来自第{best_overall_model['fold']}折, F1: {best_overall_model['f1']:.4f}, AUC: {best_overall_model['auc']:.4f}")

        # 打印最终结果
        print(f"\n=== 五折交叉验证结果 ===")
        f1_scores = [result[0] for result in fold_results]
        auc_scores = [result[1] for result in fold_results]

        for fold, (f1, auc) in enumerate(fold_results):
            print(f"第{fold + 1}折 - F1: {f1:.4f}, AUC: {auc:.4f}")

        print(f"平均 F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
        print(f"平均 AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")

        return fold_results
    
    def _reset_weights(self, m):
        """重置模型权重"""
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.LayerNorm, nn.BatchNorm1d)):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    protein_list = ['EIF4A3','EWSR1','FXR1','FXR2','IGF2BP2','IGF2BP3','MOV10'] 
    base_path = '/home/gyjsnnu/jupyterlab/CircSite/nucleotide_level_dataset/nucleotide_level_dataset/'
    
    for protein in protein_list:

        print(f"正在处理蛋白质: {protein}")
        # 加载数据
        train_data = np.load(f'processed_data/{protein}/hebing_{protein}_train.npz', allow_pickle=True)
        test_data = np.load(f'processed_data/{protein}/hebing_{protein}_test.npz', allow_pickle=True)
        print("加载hebing数据")

        all_sequences = list(train_data['sequences']) + list(test_data['sequences'])
        all_labels = list(train_data['labels']) + list(test_data['labels'])

        print(f"总序列数: {len(all_sequences)}")
        print(f"序列长度范围: {min(len(seq) for seq in all_sequences)} - {max(len(seq) for seq in all_sequences)}")

        # 创建无填充GLU模型
        model = NoPaddingGLUPredictor(
            input_dim=6,
            hidden_dims=[128, 256, 128],
            num_blocks=3
        ).to(device)

        print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

        # 运行交叉验证
        trainer = NoPaddingGLUTrainer(model, protein, device, num_folds=5)
        results = trainer.run_cross_validation(
            all_sequences, all_labels, 
            batch_size=5,  # 由于无填充，可以使用更大的batch_size
            epochs=100,
            lr=0.001
        )
    
        print("计算指标：")   
        auc = calculate_AUC(f'{protein}', 'magenta', 1)
        aupr = calculate_AUPR(f'{protein}', 'magenta', 1)
        ACC, Precision, Recall, Fscore1, MCC = calculate_metric(protein)
        
            # 定义指标文件名
        metrics_filename = 'metrics.csv'

        # 将指标保存到文件（如果文件不存在会创建，存在则追加）
        with open(metrics_filename, 'a') as f:
            f.write(f'{protein},{auc:.4f},{aupr:.4f},{ACC:.4f},{Precision:.4f},{Recall:.4f},{Fscore1:.4f},{MCC:.4f}\n')

        print(f'{protein} 指标已保存到 {metrics_filename}')