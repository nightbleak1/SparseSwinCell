
# SparseSwinCell 消融实验指南

本指南介绍如何对 SparseSwinCell 模型进行消融实验，以评估各个核心组件的贡献。

## 实验设计

我们设计了以下 5 种消融实验配置：

| 配置名称 | 说明 | 移除的组件 |
|---------|------|-----------|
| `full_model` | 完整模型（基准） | 无 |
| `no_sparse_attention` | 无稀疏注意力 | 稀疏注意力机制 |
| `no_shape_stream` | 无形状流 | 边界检测分支 |
| `no_aspp` | 无 ASPP | 空洞空间金字塔池化 |
| `no_attention_gates` | 无注意力门 | 注意力门机制 |

## 使用方法

### 1. 查看所有可用的消融配置

```bash
python cell_segmentation/trainer/ablation_study.py --list
```

### 2. 运行单个消融实验

```bash
# 运行完整模型（基准）
python cell_segmentation/trainer/ablation_study.py --ablation full_model

# 运行无稀疏注意力的实验
python cell_segmentation/trainer/ablation_study.py --ablation no_sparse_attention

# 运行无形状流的实验
python cell_segmentation/trainer/ablation_study.py --ablation no_shape_stream

# 运行无 ASPP 的实验
python cell_segmentation/trainer/ablation_study.py --ablation no_aspp

# 运行无注意力门的实验
python cell_segmentation/trainer/ablation_study.py --ablation no_attention_gates
```

### 3. 分析消融实验结果

运行完所有实验后，使用分析脚本生成报告：

```bash
python cell_segmentation/trainer/analyze_ablation_results.py
```

## 输出说明

### 训练输出

每个消融实验会在 `./logs/` 目录下创建一个独立的日志文件夹，命名格式为：
```
ablation_{配置名}_{时间戳}/
```

每个日志文件夹包含：
- `ablation_results.txt` - 实验结果摘要
- `best_checkpoint.pth` - 最佳模型权重
- `checkpoint_epoch_*.pth` - 各个 epoch 的检查点
- 训练日志和可视化图像

### 分析输出

运行分析脚本后，会在 `./ablation_analysis/` 目录下生成：
- `ablation_bpq_comparison.png` - bPQ 得分对比柱状图
- `ablation_relative_performance.png` - 相对性能对比图
- `ablation_summary_report.txt` - 详细的总结报告

## 实验结果解读

### 关键指标

- **bPQ (binary Panoptic Quality)**: 主要评估指标，衡量二值分割质量
- **Dice**: Dice 系数，衡量语义分割重叠度
- **Boundary F1**: 边界 F1 分数，衡量边界检测质量
- **mPQ**: 多类 Panoptic Quality

### 组件贡献分析

通过对比完整模型与各消融配置的性能差异，可以评估每个组件的重要性：

1. **性能下降最大** → 该组件对模型性能至关重要
2. **性能下降较小** → 该组件有帮助但非必需
3. **性能不变或提升** → 该组件可能过拟合或可以简化

## 注意事项

1. **计算资源**: 每个实验可能需要数小时到数天的训练时间，建议使用 GPU
2. **实验顺序**: 建议先运行 `full_model` 作为基准，再运行其他配置
3. **数据准备**: 确保 PanNuke 数据集已正确预处理并放置在指定路径
4. **随机种子**: 为了可重复性，建议设置固定的随机种子

## 批量运行脚本

如果你想自动运行所有消融实验，可以创建一个批量运行脚本：

```bash
#!/bin/bash
# run_all_ablation.sh

ABLATIONS=("full_model" "no_sparse_attention" "no_shape_stream" "no_aspp" "no_attention_gates")

for ablation in "${ABLATIONS[@]}"; do
    echo "Running ablation: $ablation"
    python cell_segmentation/trainer/ablation_study.py --ablation "$ablation"
done

echo "All ablations completed!"
python cell_segmentation/trainer/analyze_ablation_results.py
```

使用方法：
```bash
chmod +x run_all_ablation.sh
./run_all_ablation.sh
```

## 故障排除

### 问题：找不到数据集
确保数据集路径正确：
```python
pannuke_path = Path("/hy-tmp/SparseSwinCell/cell_segmentation/datasets/process/PanNuke")
```

### 问题：内存不足
减小 batch size：在 `ablation_study.py` 中修改：
```python
batch_size=8  # 从 16 减小到 8
```

### 问题：训练时间过长
减少 epoch 数量或使用早停机制（已默认启用）

## 引用

如果你在研究中使用了这些消融实验，请参考相关文献。

