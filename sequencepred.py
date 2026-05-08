import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
import os
from roc1 import *
from metrics1 import *
from AUPR1 import *

# 你的模型类保持不变（GatedLinearUnit, MultiScaleGLUBlock, AdaptiveSequenceProcessor, NoPaddingGLUPredictor）
class GatedLinearUnit(nn.Module):
    """门控线性单元 (GLU)"""
    def __init__(self, input_dim, output_dim):
        super(GatedLinearUnit, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim * 2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
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
        
        self.fusion_gate = GatedLinearUnit(hidden_dims[1] * len(dilations), hidden_dims[1])
        self.output_proj = nn.Linear(hidden_dims[1], hidden_dims[1])
        self.layer_norm = nn.LayerNorm(hidden_dims[1])
        
    def forward(self, x, seq_len):
        """
        输入 x 的形状应为: [batch_size, channels, sequence_length]
        """
        # 确保输入是3D的
        if x.dim() == 4:
            # 如果是 [1, batch_size, channels, seq_len]，压缩第一个维度
            if x.shape[0] == 1:
                x = x.squeeze(0)
            else:
                raise ValueError(f"意外的4D输入形状: {x.shape}")
        
        if x.dim() != 3:
            raise ValueError(f"Conv1d期望3D输入，但得到: {x.dim()}D, 形状: {x.shape}")
        
        scale_outputs = []
        
        for i, conv_block in enumerate(self.conv_blocks):
            conv_out = conv_block(x)
            conv_out = conv_out[:, :, :seq_len]
            scale_outputs.append(conv_out)
        
        fused = torch.cat(scale_outputs, dim=1)  # [batch_size, channels*dilations, seq_len]
        fused = fused.transpose(1, 2)  # [batch_size, seq_len, channels*dilations]
        fused = self.fusion_gate(fused)
        fused = self.output_proj(fused)
        fused = self.layer_norm(fused)
        
        return fused  # [batch_size, seq_len, hidden_dim]

class AdaptiveSequenceProcessor(nn.Module):
    """自适应序列处理器 - 核心GLU架构（序列级别输出）"""
    def __init__(self, input_dim=6, hidden_dims=[128, 256, 128], num_blocks=3):
        super(AdaptiveSequenceProcessor, self).__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        self.blocks = nn.ModuleList()
        
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
            
            if i < num_blocks - 1:
                dim_change = nn.Linear(hidden_dims[i], hidden_dims[i+1])
                self.blocks.append(dim_change)
        
        # 序列级别输出层
        self.sequence_pool = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]//2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[-1]//2, 1)
        )
        
    def forward(self, sequence, seq_len):
        # sequence 形状: [batch_size, seq_len, input_dim]
        x = self.input_proj(sequence)  # [batch_size, seq_len, hidden_dim[0]]
        
        block_idx = 0
        while block_idx < len(self.blocks):
            if isinstance(self.blocks[block_idx], nn.ModuleDict):
                block = self.blocks[block_idx]
                residual = x
                glu1_out = block['glu1'](x)
                x = block['layer_norm1'](glu1_out + residual)
                
                # 为卷积层准备输入: [batch_size, channels, seq_len]
                x_conv = x.transpose(1, 2)  # [batch_size, hidden_dim, seq_len]
                
                multi_scale_out = block['multi_scale'](x_conv, seq_len)  # [batch_size, seq_len, hidden_dim]
                
                residual = x
                glu2_out = block['glu2'](multi_scale_out)
                combined = glu2_out + multi_scale_out
                x = block['layer_norm2'](combined + residual)
                x = block['dropout'](x)
            else:
                x = self.blocks[block_idx](x)
            block_idx += 1
        
        # 序列级别池化：取所有位置的平均值
        sequence_representation = x.mean(dim=1)  # [batch_size, hidden_dim]
        sequence_prediction = self.sequence_pool(sequence_representation)  # [batch_size, 1]
        
        return sequence_prediction.squeeze(-1)  # [batch_size]

class NoPaddingGLUPredictor(nn.Module):
    """无填充的GLU预测模型（序列级别输出）"""
    def __init__(self, input_dim=6, hidden_dims=[128, 256, 128], num_blocks=3):
        super(NoPaddingGLUPredictor, self).__init__()
        self.processor = AdaptiveSequenceProcessor(input_dim, hidden_dims, num_blocks)
        
    def forward(self, sequences, seq_lens):
        all_predictions = []
        # 批量处理：一次性处理一个批次的所有序列
        batch_size = len(sequences)
        if batch_size > 0:
            # 将序列堆叠成一个批次
            max_len = max(seq.shape[0] for seq in sequences)
            batch_tensor = torch.zeros(batch_size, max_len, sequences[0].shape[1], 
                                      device=sequences[0].device, dtype=sequences[0].dtype)
            
            for i, seq in enumerate(sequences):
                batch_tensor[i, :seq.shape[0]] = seq
            
            # 一次性处理整个批次
            batch_predictions = self.processor(batch_tensor, max_len)
            return batch_predictions
        return torch.tensor([], device=sequences[0].device if sequences else torch.device('cpu'))

class CircRNADataset(Dataset):
    """circRNA数据集 - 使用sequence_label作为序列级别标签"""
    def __init__(self, sequences, sequence_labels):
        self.sequences = sequences
        self.sequence_labels = sequence_labels  # 序列级别标签
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            "sequence": torch.tensor(self.sequences[idx].astype(np.float32)),
            "label": torch.tensor([self.sequence_labels[idx]], dtype=torch.float32),  # 序列级别标签
            "seq_len": self.sequences[idx].shape[0]
        }

def collate_fn_variable_length(batch):
    sequences = [item["sequence"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])  # 序列级别标签
    seq_lens = [item["seq_len"] for item in batch]
    return {
        "sequences": sequences,
        "labels": labels,
        "seq_lens": seq_lens
    }

class NoPaddingGLUTrainer:
    """无填充GLU模型的训练器（序列级别）"""
    def __init__(self, model, RBP, device, num_folds=5):
        self.model = model
        self.RBP = RBP
        self.device = device
        self.num_folds = num_folds
        os.makedirs(f'glu_results_hebing_pos_neg_seq/{self.RBP}', exist_ok=True)
    
    def train_epoch(self, train_loader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        for batch in train_loader:
            sequences = [seq.to(self.device) for seq in batch["sequences"]]
            labels = batch["labels"].to(self.device)  # 序列级别标签
            seq_lens = batch["seq_lens"]
            
            optimizer.zero_grad()
            predictions = self.model(sequences, seq_lens)
            
            if len(predictions) > 0:
                loss = criterion(predictions, labels.squeeze())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                batch_count += 1
            
            if batch_count % 10 == 0:
                torch.cuda.empty_cache()
        
        return total_loss / batch_count if batch_count > 0 else 0
    
    def evaluate(self, data_loader):
        self.model.eval()
        all_true_labels = []
        all_pred_probs = []

        with torch.no_grad():
            for batch in data_loader:
                sequences = [seq.to(self.device) for seq in batch["sequences"]]
                labels = batch["labels"].to(self.device)
                seq_lens = batch["seq_lens"]

                predictions = self.model(sequences, seq_lens)

                if len(predictions) > 0:
                    pred_probs = predictions.cpu().numpy()
                    all_pred_probs.extend(pred_probs)
                    all_true_labels.extend(labels.squeeze().cpu().numpy())

                torch.cuda.empty_cache()

        return np.array(all_true_labels), np.array(all_pred_probs)
    
    def _save_results_with_format(self, true_labels, pred_probs, current_cv):
        # 将logits转换为概率值
        probabilities = 1 / (1 + np.exp(-pred_probs))  # Sigmoid转换

        # 保存真实标签
        true_file = f'glu_results_hebing_pos_neg_seq/{self.RBP}/{self.RBP}_true_fold_{current_cv}.csv'
        with open(true_file, 'w') as f:
            f.write('sequence_label\n')
            for label in true_labels:
                f.write(f'{int(label)}\n')

        # 保存预测概率 - 使用转换后的概率值
        pred_file = f'glu_results_hebing_pos_neg_seq/{self.RBP}/{self.RBP}_pred_fold_{current_cv}.csv'
        with open(pred_file, 'w') as f:
            f.write('prediction_probability\n')
            for prob in probabilities:  # ⬅️ 改为probabilities
                f.write(f'{prob:.6f}\n')

        # 保存合并结果
        combined_file = f'glu_results_hebing_pos_neg_seq/{self.RBP}/{self.RBP}_result_{current_cv}.csv'
        with open(combined_file, 'w') as f:
            f.write('prediction,true_label\n')
            for pred, true in zip(probabilities, true_labels):  # ⬅️ 改为probabilities
                f.write(f"{pred:.6f},{int(true)}\n")
    
    def run_cross_validation(self, all_sequences, all_sequence_labels, batch_size=8, epochs=50, lr=0.001):
        # 使用序列级别标签
        dataset = CircRNADataset(all_sequences, all_sequence_labels)
        
        kfold = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        fold_results = []
        best_fold_models = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            print(f"\n=== 第 {fold + 1} 折交叉验证 ===")

            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=batch_size, 
                                    shuffle=True, collate_fn=collate_fn_variable_length)
            val_loader = DataLoader(val_subset, batch_size=batch_size, 
                                  shuffle=False, collate_fn=collate_fn_variable_length)

            # 重置模型权重
            self.model.apply(self._reset_weights)
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
            criterion = nn.BCEWithLogitsLoss()   

            best_val_loss = float('inf')
            best_epoch = 0
            best_fold_model_state = None

            for epoch in range(epochs):
                train_loss = self.train_epoch(train_loader, optimizer, criterion)
                scheduler.step()

                if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                    self.model.eval()
                    val_loss = 0
                    with torch.no_grad():
                        for batch in val_loader:
                            sequences = [seq.to(self.device) for seq in batch["sequences"]]
                            labels = batch["labels"].to(self.device)
                            seq_lens = batch["seq_lens"]
                            predictions = self.model(sequences, seq_lens)
                            if len(predictions) > 0:
                                loss = criterion(predictions, labels.squeeze())
                                val_loss += loss.item()
                    
                    val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
                    
                    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_epoch = epoch + 1
                        best_fold_model_state = self.model.state_dict().copy()

                torch.cuda.empty_cache()

            print(f"最佳模型在epoch {best_epoch}, Val Loss: {best_val_loss:.4f}")

            if best_fold_model_state is not None:
                best_fold_models.append({
                    'fold': fold + 1,
                    'loss': best_val_loss,
                    'epoch': best_epoch,
                    'model_state_dict': best_fold_model_state
                })

            # 加载最佳模型进行最终评估
            if best_fold_model_state is not None:
                self.model.load_state_dict(best_fold_model_state)

            # 生成预测结果并保存
            final_true_labels, final_pred_probs = self.evaluate(val_loader)
            if len(final_true_labels) > 0:
                self._save_results_with_format(final_true_labels, final_pred_probs, fold + 1)
                print(f"第{fold + 1}折结果已保存，序列数量: {len(final_true_labels)}")
            
            fold_results.append(best_val_loss)

        # 保存最佳模型
        if best_fold_models:
            best_fold_models.sort(key=lambda x: x['loss'])
            best_overall_model = best_fold_models[0]

            best_model_save_path = f'glu_results_hebing_pos_neg_seq/{self.RBP}/{self.RBP}_best_overall_model.pth'
            torch.save({
                'fold': best_overall_model['fold'],
                'epoch': best_overall_model['epoch'],
                'model_state_dict': best_overall_model['model_state_dict'],
                'val_loss': best_overall_model['loss'],
                'all_fold_results': fold_results
            }, best_model_save_path)
            print(f"\n🎯 五折中最佳模型已保存到: {best_model_save_path}")
            print(f"   来自第{best_overall_model['fold']}折, Loss: {best_overall_model['loss']:.4f}")

        print(f"\n=== 五折交叉验证结果 ===")
        for fold, loss in enumerate(fold_results):
            print(f"第{fold + 1}折 - Loss: {loss:.4f}")
        print(f"平均 Loss: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")

        return fold_results
    
    def _reset_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.LayerNorm, nn.BatchNorm1d)):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

def load_pos_neg_data(protein_name):
    """
    加载正负样本数据 - 修复字段名匹配问题
    """
    data_dir = f'processed_data/{protein_name}'
    
    # 尝试加载正样本数据
    positive_files = [
        os.path.join(data_dir, f'hebing_{protein_name}_merged.npz')
    ]
    
    positives_data = None
    positives_file = None
    
    for file_path in positive_files:
        if os.path.exists(file_path):
            positives_data = np.load(file_path, allow_pickle=True)
            positives_file = file_path
            print(f"加载正样本数据: {file_path}")
            
            # 调试：显示正样本文件中的所有字段
            print(f"正样本文件字段: {positives_data.files}")
            
            # 检查有哪些标签字段
            available_fields = positives_data.files
            print(f"可用的标签字段: {[f for f in available_fields if 'label' in f.lower() or 'label' in f]}")
            
            break
    
    if positives_data is None:
        raise FileNotFoundError(f"未找到正样本文件，尝试了: {positive_files}")
    
    # 尝试加载负样本数据
    negatives_file = os.path.join(data_dir, f'hebing_{protein_name}_negative.npz')
    if not os.path.exists(negatives_file):
        raise FileNotFoundError(f"未找到负样本文件: {negatives_file}")
    
    print(f"加载负样本数据: {negatives_file}")
    negatives_data = np.load(negatives_file, allow_pickle=True)
    
    # 调试：显示负样本文件中的所有字段
    print(f"负样本文件字段: {negatives_data.files}")
    
    # 提取序列
    if 'sequences' not in positives_data:
        raise KeyError("正样本数据中未找到'sequences'字段")
    
    pos_sequences = list(positives_data['sequences'])
    
    # 读取序列级别标签 - 尝试多种可能的字段名
    pos_sequence_labels = None
    
    # 按优先级尝试不同的字段名
    possible_label_fields = [
        'sequence_labels'    # 你的实际字段名（复数）
    ]
    
    for field in possible_label_fields:
        if field in positives_data:
            pos_sequence_labels = list(positives_data[field])
            print(f"  正样本使用 '{field}' 字段作为序列级别标签")
            break
    
    if pos_sequence_labels is None:
        raise KeyError(f"正样本数据中未找到序列级别标签字段。可用的字段: {positives_data.files}")
    
    # 负样本数据
    if 'sequences' not in negatives_data:
        raise KeyError("负样本数据中未找到'sequences'字段")
    
    neg_sequences = list(negatives_data['sequences'])
    
    # 读取负样本的序列级别标签
    neg_sequence_labels = None
    
    for field in possible_label_fields:
        if field in negatives_data:
            neg_sequence_labels = list(negatives_data[field])
            print(f"  负样本使用 '{field}' 字段作为序列级别标签")
            break
    
    if neg_sequence_labels is None:
        raise KeyError(f"负样本数据中未找到序列级别标签字段。可用的字段: {negatives_data.files}")
    
    # 合并正负样本
    all_sequences = pos_sequences + neg_sequences
    all_sequence_labels = pos_sequence_labels + neg_sequence_labels
    
    # 验证数据一致性
    if len(all_sequences) != len(all_sequence_labels):
        raise ValueError(f"数据数量不匹配: 序列{len(all_sequences)}个, 标签{len(all_sequence_labels)}个")
    
    # 验证标签值
    unique_labels = np.unique(all_sequence_labels)
    print(f"数据统计:")
    print(f"  正样本数: {len(pos_sequences)}")
    print(f"  负样本数: {len(neg_sequences)}")
    print(f"  总样本数: {len(all_sequences)}")
    print(f"  唯一标签值: {unique_labels}")
    
    # 计算标签分布
    positive_count = sum(1 for label in all_sequence_labels if label == 1)
    negative_count = sum(1 for label in all_sequence_labels if label == 0)
    other_count = len(all_sequence_labels) - positive_count - negative_count
    
    print(f"  标签分布: 1={positive_count}, 0={negative_count}")
    if other_count > 0:
        print(f"  其他标签: {other_count}")
    
    # 特征维度检查
    if pos_sequences and len(pos_sequences) > 0:
        # 检查序列的维度
        sample_seq = pos_sequences[0]
        if hasattr(sample_seq, 'shape'):
            print(f"  特征维度: {sample_seq.shape[1]}维 (序列长度: {sample_seq.shape[0]})")
        else:
            print(f"  序列类型: {type(sample_seq)}")
    
    return all_sequences, all_sequence_labels

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    #protein_list = ['EIF4A3','EWSR1','FXR1','FXR2','IGF2BP2','IGF2BP3','MOV10']
    protein_list = ['FXR1','FXR2','IGF2BP2','IGF2BP3']
    # 创建结果目录
    os.makedirs('glu_results_hebing_pos_neg_seq', exist_ok=True)
    
    # 创建指标文件（如果不存在）
    metrics_filename = 'metrics_pos_neg_sequence_level.csv'
    if not os.path.exists(metrics_filename):
        with open(metrics_filename, 'w') as f:
            f.write('Protein,AUC,AUPR,ACC,Precision,Recall,F1_Score,MCC\n')
    
    for protein in reversed(protein_list):
        print(f"\n{'='*80}")
        print(f"正在处理蛋白质: {protein}")
        print(f"{'='*80}")
        
        try:
            # 加载数据（使用序列级别标签）
            all_sequences, all_sequence_labels = load_pos_neg_data(protein)
            
            print(f"序列长度范围: {min(len(seq) for seq in all_sequences)} - {max(len(seq) for seq in all_sequences)}")
            print(f"特征维度: {all_sequences[0].shape[1]}维")
            
            # 创建模型
            model = NoPaddingGLUPredictor(
                input_dim=6,
                hidden_dims=[128, 256, 128],
                num_blocks=3
            ).to(device)
            
            print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
            
            # 运行交叉验证
            trainer = NoPaddingGLUTrainer(model, protein, device, num_folds=5)
            results = trainer.run_cross_validation(
                all_sequences, all_sequence_labels, 
                batch_size=8,
                epochs=100,
                lr=0.001
            )
            
            # 使用你的计算函数计算指标
            print("\n使用计算函数计算指标...")
            auc = calculate_AUC(protein, 'magenta', 1)
            aupr = calculate_AUPR(protein, 'magenta', 1)
            ACC, Precision, Recall, Fscore1, MCC = calculate_metric(protein)
            
            # 保存指标
            with open(metrics_filename, 'a') as f:
                f.write(f'{protein},{auc:.4f},{aupr:.4f},{ACC:.4f},{Precision:.4f},{Recall:.4f},{Fscore1:.4f},{MCC:.4f}\n')
            
            print(f'{protein} 指标已保存到 {metrics_filename}')
            
            # 清理内存
            del model, trainer, all_sequences, all_sequence_labels
            torch.cuda.empty_cache()
            
        except FileNotFoundError as e:
            print(f"错误: {e}")
            print(f"跳过 {protein}...")
            continue
        except KeyError as e:
            print(f"数据字段错误: {e}")
            print(f"跳过 {protein}...")
            continue
        except Exception as e:
            print(f"处理 {protein} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print("所有蛋白质处理完成!")
    print(f"序列级别结果保存在 glu_results_hebing_pos_neg_seq/ 目录")
    print(f"序列级别指标保存在 {metrics_filename}")
    print(f"{'='*80}")
