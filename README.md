# SparseSwinCell: High-Performance Cell Segmentation with Sparse Swin Transformers

**SparseSwinCell** 是一个先进的病理图像分析系统，专为复杂的细胞核分割与分类任务设计。它基于 **Swin Transformer V2** 骨干网络，创新性地引入了 **稀疏注意力机制 (Sparse Attention)** 和 **多任务形状流 (Multi-task Shape Stream)**，在保持高效计算的同时，显著提升了对粘连细胞和微小细胞的检测精度。

本项目针对 **PanNuke** 和 **CoNSeP** 等高难度数据集进行了深度优化，集成了从数据预处理、增强、训练到全切片推理 (WSI Inference) 的完整工作流。

---

## 🌟 核心创新 (Key Innovations)

1.  **稀疏 Swin Transformer (Sparse Attention)**
    *   引入 **KM Attention** 机制，基于熵 (Entropy) 动态调整注意力保留比例 (Top-K)。
    *   大幅降低 Transformer 的计算复杂度，同时保留关键的上下文依赖。

2.  **双流架构与形状感知 (Shape Stream)**
    *   除了主干网络，设计了独立的 **Shape Stream** 分支，专门提取细胞边缘特征。
    *   通过 **Attention Gates** 将边缘信息注入解码器，显著改善了细胞边界的分割效果。

3.  **四阶段渐进式训练策略 (4-Stage Progressive Training)**
    *   独创的训练调度方案，从全局特征学习逐步过渡到边缘攻坚。
    *   **动态边界损失**: 在训练后期根据验证集指标自动提升边界损失权重，强制模型关注困难样本。

4.  **全方位数据增强与归一化**
    *   集成 **Reinhard 染色归一化** 和 **局部相位增强**，解决不同实验室制片的色差问题。
    *   使用 **WeightedRandomSampler** 和 **动态损失加权** 彻底解决类别不平衡（如 Dead/Connective 细胞稀缺）问题。

---

## 🏗 模型架构 (Model Architecture)

SparseSwinCell 采用改进的 U-Net 编码器-解码器结构，包含以下关键组件：

### 1. 编码器 (Encoder)
*   **Backbone**: Swin Transformer V2 (Tiny/Small/Base)。
*   **Sparse Mechanism**:
    *   **Dynamic K**: 根据注意力图的熵值自动计算最优 K 值。
    *   **Head Pruning**: 可学习的注意力头权重，自动抑制冗余的 Head。

### 2. 颈部 (Neck)
*   **ASPP**: 空洞空间金字塔池化 (Dilations: 1, 6, 12, 18)，捕获多尺度上下文。
*   **Shape Stream**: 并行卷积分支，专注于生成高频边缘特征 (`edge_map`)。

### 3. 解码器 (Decoder) - 多任务输出
采用多分支结构，通过 **Attention Gates** 融合编码器特征：
*   **Nuclei Binary Branch**: 预测细胞核前景/背景 (Focal Tversky Loss)。
*   **HV Map Branch**: 预测水平/垂直距离图，用于分离粘连细胞 (MSE + MSGE Loss)。
    *   融合了 Shape Stream 的特征以增强边界。
*   **Nuclei Type Branch**: 预测 6 类细胞核类型 (Neoplastic, Inflammatory, etc.)。
*   **Tissue Type Branch**: 基于全局特征 (GAP) 预测 19 类组织来源。

---

## 🚀 训练策略 (Training Strategy)

这是本项目取得高性能的关键。我们采用精细的 **"先粗后细，重点攻坚"** 策略。

### 1. 损失函数体系
总损失为各任务加权和：
*   **分割**: Focal Tversky Loss (1.0) + Dice Loss (1.0)
*   **回归 (HV Map)**: MSE Loss (1.0) + **MSGE Loss (Gradient Consistency)** (1.0)
*   **分类**: Cross Entropy + Dice + Focal Loss
*   **边界**: **Dynamic Boundary Loss** (初始 0.5 -> 最高 1.7)

### 2. 优化器与调度
*   **Optimizer**: AdamW (LR=1e-4, Weight Decay=1e-5).
*   **Scheduler**:
    *   **Epoch 0-90**: Cosine Annealing (1e-4 -> 1e-6).
    *   **Epoch 90 (Hard Restart)**: 强制重置 LR 为 **2e-5**，跳出局部最优。
    *   **Epoch 90-120**: Linear Decay (2e-5 -> 5e-6)，进行最后微调。

### 3. 四阶段权重调整 (Four-Stage Weight Adjustment)
| 阶段 | Epoch | 策略描述 | 关键调整 |
| :--- | :--- | :--- | :--- |
| **I** | 0-50 | **基础学习** | 各任务权重平衡，学习通用特征。 |
| **II** | 50-60 | **核心强化** | 稳步强化核心损失，温和弱化非核心损失。 |
| **III** | 60-90 | **动态适配** | 维持分类权重，防止 mPQ 崩溃；引入动态边界监控。 |
| **IV** | 90+ | **边界攻坚** | **边界损失权重 -> 1.5+** (若 F1 无提升则 +0.1) <br> **边缘流权重 -> 2.0** <br> **分类权重 -> 2.0** |

---

## 🛠 安装与使用 (Installation & Usage)

### 环境准备
```bash
conda create -n sparseswincell python=3.11
conda activate sparseswincell
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

### 数据准备
```bash
# 预处理 PanNuke 数据集 (生成 HV Maps, 归一化等)
python cell_segmentation/datasets/prepare_pannuke.py
```

### 训练 (Training)
```bash
# 从头开始训练 (推荐)
python cell_segmentation/trainer/train_from_scratch.py

# 从断点恢复
python cell_segmentation/trainer/train_from_scratch.py --resume --checkpoint logs/latest_checkpoint.pth
```

### 评估 (Evaluation)
```bash
# 评估 PanNuke 数据集 (自动读取 config)
python cell_segmentation/inference/inference_cellvit_experiment_pannuke.py \
    --run_dir logs/train_cellvit_timestamp \
    --gpu 0

# 评估 CoNSeP 数据集
python cell_segmentation/inference/inference_cellvit_experiment_consep.py \
    --run_dir logs/train_cellvit_timestamp \
    --checkpoint_name final_model.pth \
    --gpu 0 \
    --magnification 40 \
    --plots
```

### 全切片推理 (WSI Inference)
支持多进程处理巨大的病理切片 (.svs, .tiff 等)：
```bash
# 处理整个 WSI 数据集
python cell_segmentation/inference/cell_detection_mp.py process_dataset \
    --model checkpoints/model_best.pth \
    --wsi_paths /path/to/wsi_folder \
    --output_path /path/to/output \
    --n_postprocess_workers 8
```

---

## 🔬 消融实验 (Ablation Study)

为了验证各个核心组件的贡献，我们提供了完整的消融实验框架，可以系统地评估每个模块对性能的影响。

### 实验配置
| 配置名称 | 说明 | 移除的组件 |
|---------|------|-----------|
| `full_model` | 完整模型（基准） | 无 |
| `no_sparse_attention` | 无稀疏注意力 | 稀疏注意力机制 |
| `no_shape_stream` | 无形状流 | 边界检测分支 |
| `no_aspp` | 无 ASPP | 空洞空间金字塔池化 |
| `no_attention_gates` | 无注意力门 | 注意力门机制 |

### 运行消融实验

#### 1. 查看可用配置
```bash
python cell_segmentation/trainer/ablation_study.py --list
```

#### 2. 运行单个消融实验
```bash
# 运行无稀疏注意力的实验
python cell_segmentation/trainer/ablation_study.py --ablation no_sparse_attention

# 运行无形状流的实验
python cell_segmentation/trainer/ablation_study.py --ablation no_shape_stream

# 运行无 ASPP 的实验
python cell_segmentation/trainer/ablation_study.py --ablation no_aspp

# 运行无注意力门的实验
python cell_segmentation/trainer/ablation_study.py --ablation no_attention_gates
```

#### 3. 批量运行所有消融实验
```bash
./run_ablation_experiments.sh
```

#### 4. 分析实验结果
```bash
python cell_segmentation/trainer/analyze_ablation_results.py
```

### 消融实验输出
每个实验会在 `./logs/` 目录下生成独立的日志文件夹，包含：
- 训练检查点
- 实验结果摘要 (`ablation_results.txt`)
- 可视化图像

运行分析脚本后，会在 `./ablation_analysis/` 目录生成：
- 性能对比柱状图
- 相对性能对比图
- 详细总结报告 (`ablation_summary_report.txt`)

详细使用指南请参考 [ABLATION_STUDY_GUIDE.md](./ABLATION_STUDY_GUIDE.md)

---

## 📊 性能指标 (Metrics)

模型在测试时会自动计算以下指标：
*   **Instance Segmentation**: Panoptic Quality (PQ), bPQ (Binary), mPQ (Multi-class).
*   **Detection**: F1-Score, Precision, Recall.
*   **Semantic Segmentation**: Dice, Jaccard (IoU).
*   **Boundary Quality**: Boundary F1-Score.

---

## 📂 项目结构

```text
SparseSwinCell/
├── base_ml/                           # 基础训练框架
├── cell_segmentation/
│   ├── datasets/                      # 数据集加载与预处理 (PanNuke, CoNSeP)
│   ├── inference/                     # 推理脚本 (WSI, Experiment Evaluation)
│   ├── models/                        # 模型定义
│   │   ├── backbone/                  # Swin Transformer V2
│   │   ├── cellvit.py                 # 基础模型结构
│   │   └── sparse_cellvit.py          # 稀疏化与多任务实现
│   ├── trainer/                       # 训练逻辑
│   │   ├── train_from_scratch.py     # 主训练脚本
│   │   ├── ablation_study.py         # 消融实验训练脚本
│   │   └── analyze_ablation_results.py # 消融实验结果分析
│   └── utils/                         # 指标计算与后处理
├── configs/                           # 实验配置文件
├── logs/                              # 训练日志与模型权重
├── ablation_analysis/                 # 消融实验分析结果（运行后生成）
├── ABLATION_STUDY_GUIDE.md           # 消融实验详细指南
└── run_ablation_experiments.sh        # 批量运行消融实验脚本
```

## 📝 许可证与引用

本项目采用 MIT 许可证。
参考论文：
1. Swin Transformer V2 (arXiv:2111.09883)
2. HoVer-Net (arXiv:1812.06499)
3. CellViT (arXiv:2306.15350)

---
如有问题，请联系: [1776535661@qq.com]
