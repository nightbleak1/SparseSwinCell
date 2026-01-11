# SparseSwinCell: Cell Segmentation with Sparse Swin Transformers

## 项目概述

SparseSwinCell是一个基于Vision Transformer的细胞分割系统，以Swin Transformer V2为backbone，并融入了多种稀疏VIT机制。该系统能够准确分割组织切片图像中的细胞核，并识别其类型，支持多种数据集，包括PanNuke和MoNuSeg。

项目包含完整的训练、评估和推理流程，支持从头训练、断点续训和超参数扫描，提供了多种模型变体以适应不同的应用场景。

## 技术栈

- **深度学习框架**: PyTorch
- **计算机视觉库**: torchvision, OpenCV
- **模型架构**: Swin Transformer V2, SparseSwinCell, 稀疏VIT
- **数据集**: PanNuke, MoNuSeg
- **损失函数**: BCEWithLogitsLoss, Focal Loss
- **优化器**: AdamW
- **学习率调度**: CosineAnnealingLR
- **混合精度训练**: AMP

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
- **细胞核类型映射**: 预测每个细胞核的类型（6通道）
- **组织类型预测**: 基于全局特征预测图像的组织类型（19通道）
  - 增强型分类头架构，包含BatchNorm、GELU激活和dropout层
  - 改进的特征处理，添加了dropout层提高泛化能力
  - 更高的损失权重，确保模型充分关注组织分类任务

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
    - 组织分类损失权重: 2.0（提高模型对组织分类的关注度）
- **优化器**: AdamW，初始学习率1e-4
- **学习率调度**: CosineAnnealingLR
- **批量大小**: 32（可根据GPU内存调整）
- **混合精度训练**: FP16
- **早期停止**: 监控验证指标，当性能不再提升时停止训练
- **CUDA优化**: 
  - 启用cuDNN自动调优
  - 增加workers数量，加速数据加载
  - 启用persistent_workers和prefetch_factor，优化数据预取

### 从头训练

使用`train_from_scratch.py`脚本进行从头训练：

```bash
cd /hy-tmp/SparseSwinCell && python cell_segmentation/trainer/train_from_scratch.py
```

### 断点续训

支持从检查点恢复训练，自动保存最佳模型、最新模型和定期检查点。

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
6. **CUDA优化**: 
    - 启用cuDNN自动调优，加速卷积运算
    - 增加workers数量，提高数据加载速度
    - 启用persistent_workers和prefetch_factor，优化数据预取
    - 可根据GPU内存调整批量大小

## 评估指标

模型使用以下指标进行评估：

- **细胞核分割**: IoU, F1分数
- **细胞核类型分类**: 准确率, F1分数
- **实例分割**: PQ (Panoptic Quality), AP (Average Precision)
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