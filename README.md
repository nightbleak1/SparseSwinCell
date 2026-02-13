# SparseSwinCell: Cell Segmentation with Sparse Swin Transformers

## 项目概述

SparseSwinCell是一个基于Vision Transformer的细胞分割系统，以Swin Transformer V2为backbone，并融入了多种稀疏VIT机制。该系统能够准确分割组织切片图像中的细胞核，并识别其类型，支持多种数据集，包括PanNuke和MoNuSeg。

项目包含完整的训练、评估和推理流程，支持从头训练、断点续训和超参数扫描，提供了多种模型变体以适应不同的应用场景。

## 技术栈

- **深度学习框架**: PyTorch
- **计算机视觉库**: torchvision, OpenCV, albumentations
- **模型架构**: Swin Transformer V2, SparseSwinCell, 稀疏VIT
- **数据集**: PanNuke, MoNuSeg
- **损失函数**: BCEWithLogitsLoss, Focal Loss, MSELoss, 边界感知损失
- **优化器**: AdamW
- **学习率调度**: CosineAnnealingLR
- **混合精度训练**: AMP
- **评估指标**: ARI, IoU, Boundary F1-Score, Macro-F1, Weighted-F1, PQ

## 模型架构

### 核心模型

SparseSwinCell采用编码器-解码器架构，主要包含以下组件：

#### 1. Swin Transformer Backbone with km Attention
- **Patch Embedding**: 将输入图像分割为4x4的补丁，线性投影到嵌入空间
- **Swin Transformer Layers**: 4个阶段的分层设计，每个阶段包含多个Swin Transformer层
- **km Attention**: 只保留top-k比例的注意力权重，减少计算复杂度
- **Patch Merging**: 每个阶段结束时进行下采样，特征图尺寸减半，通道数翻倍

#### 2. 解码器结构
- **瓶颈上采样**: 将编码器输出的最高级特征上采样
- **多级融合**: 依次与编码器各阶段输出的特征进行融合
- **反卷积块**: 使用Deconv2DBlock进行上采样和特征提取
- **卷积块**: 使用Conv2DBlock进行特征融合和处理

#### 3. 多分支输出
- **细胞核二进制映射**: 预测每个像素是否为细胞核（2通道）
- **HV映射**: 预测每个细胞核像素的水平和垂直偏移（2通道），用于实例分割
  - 增强型解码器，包含边界感知卷积
  - 注意力融合机制，提高边界准确性
- **细胞核类型映射**: 预测每个细胞核的类型（6通道）
  - 增强型分类头，包含通道注意力机制
  - 改进的特征融合，提高分类准确性
- **组织类型预测**: 基于全局特征预测图像的组织类型（19通道）
  - 增强型分类头架构，包含BatchNorm、GELU激活和dropout层
  - 改进的特征处理，添加了dropout层提高泛化能力
  - 全局平均池化（GAP）层，捕获全局上下文信息

### 模型变体

项目包含多个模型变体，位于`models/segmentation/cell_segmentation/`目录：
- **cellvit.py**: 基础SparseSwinCell模型，基于Swin Transformer with km Attention
- **sparse_cellvit.py**: 增强型稀疏CellViT模型，包含更多稀疏机制
- **sparse_utils.py**: 稀疏VIT相关工具函数
- **utils.py**: 通用工具函数

### 稀疏VIT机制

项目实现了多种稀疏VIT机制，用于优化模型性能和效率：
- **分层稀疏性**: 在不同阶段采用不同的稀疏策略
- **基于内容的稀疏性**: 根据特征内容动态调整注意力权重
- **动态km attention**: 根据输入内容动态调整top-k比例
- **全局稀疏性**: 跨层的稀疏策略协调

### 模型复杂度

- **总参数量 (Total Parameters)**: 97.89 M
- **可训练参数 (Trainable Parameters)**: 97.89 M
- **计算量 (FLOPs)**: 57.74 G (每张 256x256 输入图像)

## 数据处理

### 数据集划分

- **PanNuke**: 
  - 训练集: fold0, fold1
  - 测试集: fold2
- **MoNuSeg**: 用于验证（训练时不使用，仅在训练结束后用于最终模型评估）

### 染色归一化

为了处理不同组织切片之间的染色差异，项目实现了基于风格一致性的染色归一化方法：
- **局部相位增强**: 增强图像的相位信息，突出细胞核结构
- **风格一致性熵模型**: 确保不同图像之间的染色风格一致
- **CDF归一化**: 对每个颜色通道进行累积分布函数归一化
- **Lab颜色空间处理**: 在Lab空间中进行相位处理，然后转换回RGB

### 数据增强

支持多种数据增强操作：
- 随机翻转
- 随机旋转
- 随机缩放
- 颜色扰动

## 训练流程

### 训练配置

- **损失函数**: 
  - BCEWithLogitsLoss（用于二进制分割任务）
  - CrossEntropyLoss（用于细胞核类型和组织类型分类）
  - MSELoss（用于HV映射预测）
  - Focal Loss（用于细胞核类型分类，处理类别不平衡）
  - 边界感知损失（用于HV映射预测，增强边界准确性）
- **优化器**: AdamW，初始学习率1e-4
- **学习率调度**: CosineAnnealingLR
- **批量大小**: 18（可根据GPU内存调整，支持梯度累积）
- **混合精度训练**: FP16
- **早期停止**: 监控验证指标，当性能不再提升时停止训练
- **CUDA优化**: 
  - 启用cuDNN自动调优
  - 增加workers数量，加速数据加载
  - 启用persistent_workers和prefetch_factor，优化数据预取
  - 启用TF32加速，提高计算速度
- **权重调整策略**: 四阶段渐进式调整
  - 第一阶段（50-60 epoch）：稳步强化核心损失，温和弱化非核心损失
  - 第二阶段（60-70 epoch）：暂停核心损失大幅调整，针对性强化边界
  - 第三阶段（70-90 epoch）：核心损失达目标后稳定，动态适配分类指标
  - 第四阶段（90轮后）：聚焦边界损失，大幅降低其他损失权重
    - 边界损失：1.5（基础值）→ 连续 3 轮无提升则调至 1.6（最高 1.7）
    - nuclei_binary_map (BCE)：1.0（恢复实例分割基础）
    - hv_map (MSE)：1.0（提升辅助分割权重）
    - nuclei_type_map (CE)：2.0（保持高分类权重）
    - tissue_types (CE)：1.0（提升组织分类权重，提供语义上下文）
    - edge_map (Shape Stream)：2.0（进一步强化边缘）
    - 学习率：Epoch 100-120 进行线性衰减 (Linear Decay) 从 1e-5 降至 5e-6
    - **特殊处理 (Dead/Connective)**: 
      - Dead 细胞权重强制设为 5.0，Connective 细胞权重强制设为 1.5
      - 组织类型 Loss 根据逆频率自动加权（如 Kidney ~3.1, Breast ~0.18）
- **Focal Loss动态调整**: 根据验证集Macro-F1变化自动调整权重
- **时间戳日志目录**: 每次训练自动创建唯一的时间戳日志目录，避免日志冲突

### 从头训练

使用`train_from_scratch.py`脚本进行从头训练：

```bash
cd /hy-tmp/SparseSwinCell && python cell_segmentation/trainer/train_from_scratch.py
```

### 断点续训

支持从检查点恢复训练，自动保存最佳模型、最新模型和定期检查点。

#### 从特定检查点恢复训练

```bash
# 从第50轮检查点恢复训练
cd /hy-tmp/SparseSwinCell && python cell_segmentation/trainer/train_from_scratch.py --resume --checkpoint logs/train_cellvit_from_scratch/checkpoint_epoch_50.pth

# 从最近的断点恢复训练
cd /hy-tmp/SparseSwinCell && python cell_segmentation/trainer/train_from_scratch.py --resume --checkpoint logs/train_cellvit_20260203_010235/latest_checkpoint.pth
```

## 项目结构

```
SparseSwinCell/
├── base_ml/                   # 基础机器学习组件
│   └── base_trainer.py        # 基础训练器
├── cell_segmentation/         # 细胞分割主模块
│   ├── data_preparation.py    # 数据准备脚本
│   ├── datasets/              # 数据集处理
│   │   ├── base_cell.py       # 基础细胞数据集类
│   │   ├── pannuke.py         # PanNuke数据集处理
│   │   ├── monuseg.py         # MoNuSeg数据集处理
│   │   └── prepare_*.py       # 数据集预处理脚本
│   ├── evaluate.py            # 评估脚本
│   ├── experiments/           # 实验配置和结果
│   ├── inference/             # 推理相关代码
│   ├── inference.py           # 推理脚本
│   ├── models/                # 模型定义
│   │   └── backbone/          # 骨干网络
│   │       └── swin_transformer.py  # Swin Transformer定义
│   ├── pretrain_mae.py        # MAE预训练脚本
│   ├── run_sparse_cellvit.py  # 稀疏CellViT运行脚本
│   ├── trainer/               # 训练脚本
│   │   ├── trainer_cellvit.py # SparseSwinCell训练器
│   │   └── train_from_scratch.py  # 从头训练脚本
│   └── utils/                 # 工具函数
├── checkpoints/               # 模型检查点
├── configs/                   # 配置文件
├── datamodel/                 # 数据模型定义
├── docs/                      # 文档
├── example/                   # 示例代码
├── environment.yml            # Conda环境配置
├── experiment_logs/           # 实验日志
├── logs/                      # 训练日志
├── logs_paper/                # 论文相关日志
├── makefile                   # 构建脚本
├── models/                    # 模型库
│   └── segmentation/          # 分割模型
│       └── cell_segmentation/ # 细胞分割模型
│           ├── cellvit.py     # SparseSwinCell核心模型
│           ├── sparse_cellvit.py # 稀疏CellViT模型
│           ├── sparse_utils.py # 稀疏VIT相关工具函数
│           └── utils.py       # 通用工具
├── optional_dependencies.txt  # 可选依赖
├── preprocessing/             # 预处理脚本
├── reports/                   # 报告
├── requirements.txt           # 必要依赖
├── utils/                     # 通用工具函数
├── .flake8                    # Flake8配置
├── .gitignore                 # Git忽略文件
├── .pre-commit-config.yaml    # 预提交钩子配置
├── LICENSE                    # 许可证
└── README.md                  # 项目说明文档
```

## 如何使用

### 创建虚拟环境

推荐使用conda创建虚拟环境，以确保依赖包的版本兼容性：

```bash
# 创建虚拟环境
conda create -n sparseswincell python=3.11

# 激活虚拟环境
conda activate sparseswincell

# 安装PyTorch和torchvision（根据CUDA版本调整）
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 安装依赖

在激活虚拟环境后，安装项目所需的其他依赖：

```bash
pip install -r requirements.txt
```

### 数据准备

1. 下载PanNuke和MoNuSeg数据集
2. 运行预处理脚本：
   ```bash
   cd SparseSwinCell && python cell_segmentation/datasets/prepare_pannuke.py
   cd SparseSwinCell && python cell_segmentation/datasets/prepare_monuseg.py
   ```

### 训练模型

#### 使用train_from_scratch.py进行从头训练

```bash
cd SparseSwinCell && python cell_segmentation/trainer/train_from_scratch.py
```

#### 使用run_sparse_cellvit.py运行完整实验

```bash
cd SparseSwinCell

# 使用默认配置运行稀疏CellViT实验
python cell_segmentation/run_sparse_cellvit.py

# 从检查点恢复实验
python cell_segmentation/run_sparse_cellvit.py --checkpoint <path_to_checkpoint>

# 运行超参数扫描
python cell_segmentation/run_sparse_cellvit.py --run_sweep
```

### 评估模型

训练结束后，模型将自动在测试集上进行评估。可以使用以下命令手动评估：

```bash
cd SparseSwinCell

# 使用专门的评估脚本
python cell_segmentation/evaluate.py --checkpoint <path_to_checkpoint>
```

### 推理

使用训练好的模型进行推理：

```bash
cd SparseSwinCell && python cell_segmentation/inference.py --checkpoint <path_to_checkpoint> --image <path_to_image>
```

## 改进点

1. **采用Swin Transformer V2**: 使用Swin Transformer V2作为backbone，替代传统ViT
2. **稀疏VIT机制**: 加入多种稀疏注意力机制，提高模型效率
3. **染色归一化**: 实现基于风格一致性的染色归一化，处理染色差异问题
4. **删除RGB转换**: 直接使用原始RGB格式，避免不必要的转换
5. **增强组织分类**: 
   - 改进组织分类头架构，添加BatchNorm、GELU激活和dropout层
   - 增强特征处理，添加dropout层提高泛化能力
   - 全局平均池化（GAP）层，捕获全局上下文信息
6. **增强HV映射预测**: 
   - 增强型解码器，包含边界感知卷积
   - 注意力融合机制，提高边界准确性
   - 边界感知损失函数，针对性强化边界特征学习
7. **增强细胞核类型分类**: 
   - 增强型分类头，包含通道注意力机制
   - 改进的特征融合，提高分类准确性
   - Focal Loss，处理类别不平衡问题
8. **四阶段权重调整策略**: 
   - 第一阶段（50-60 epoch）：稳步强化核心损失，温和弱化非核心损失
   - 第二阶段（60-70 epoch）：暂停核心损失大幅调整，针对性强化边界
   - 第三阶段（70-90 epoch）：核心损失达目标后稳定，动态适配分类指标
   - 第四阶段（90轮后）：聚焦边界损失，大幅降低其他损失权重
9. **Focal Loss动态调整**: 根据验证集Macro-F1变化自动调整权重
10. **时间戳日志目录**: 每次训练自动创建唯一的时间戳日志目录，避免日志冲突
11. **CUDA优化**: 
    - 启用cuDNN自动调优，加速卷积运算
    - 增加workers数量，提高数据加载速度
    - 启用persistent_workers和prefetch_factor，优化数据预取
    - 启用TF32加速，提高计算速度
    - 可根据GPU内存调整批量大小
12. **类别不平衡处理 (Class Imbalance Handling)**:
    - 使用 `WeightedRandomSampler` 进行基于组织和细胞类型的平衡采样
    - 动态损失重加权 (Dynamic Loss Re-weighting)：
      - 组织类型：逆频率加权（如 Kidney ~3.1, Breast ~0.18）
      - 细胞核类型：针对稀缺类别（Dead: 5.0）和难分类别（Connective: 1.5）进行特殊强化
13. **小目标召回优化 (Small Object Recall Optimization)**:
    - 调整后处理阈值 (`min_size` 从 10 降至 5)，显著提升微小细胞（如 Dead, Connective）的召回率
14. **精细化训练策略 (Refined Training Strategy)**:
    - 延长训练周期至 120 Epoch
    - 线性学习率衰减 (Linear Decay)：Epoch 100-120 从 1e-5 降至 5e-6，确保 Loss 面底端的平稳收敛

## 评估指标

模型使用以下指标进行评估：

- **细胞核分割**: IoU, F1分数
- **细胞核类型分类**: 准确率, Macro-F1, Weighted-F1
- **实例分割**: PQ (Panoptic Quality), AP (Average Precision), ARI (Adjusted Rand Index)
- **边界准确性**: Boundary F1-Score
- **组织类型分类**: 准确率, F1分数

## 许可证

本项目采用MIT许可证。

## 参考文献

1. Swin Transformer: https://arxiv.org/abs/2103.14030
2. Swin Transformer V2: https://arxiv.org/abs/2111.09883
3. PanNuke Dataset: https://arxiv.org/abs/2003.10778
4. MoNuSeg Dataset: https://arxiv.org/abs/1806.05587

## 联系方式

如有问题或建议，请通过以下方式联系：

- 邮箱: [1776535661@qq.com]