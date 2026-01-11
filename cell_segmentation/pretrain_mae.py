import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import os

# 导入自定义模块
from cell_segmentation.models.backbone.swin_transformer import SwinTransformerWithKMAttention
from cell_segmentation.datasets.pannuke import PanNukeDataset

class ChannelAttention(nn.Module):
    """通道注意力机制"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * (avg_out + max_out) * 0.5

class SpatialAttention(nn.Module):
    """空间注意力机制"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        return x * self.sigmoid(attention)

class CBAM(nn.Module):
    """通道与空间注意力模块"""
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attn = ChannelAttention(in_channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x

class MultiScaleFusionDecoder(nn.Module):
    """多尺度特征融合解码器"""
    def __init__(self, feature_dims, out_channels=2):
        super().__init__()
        self.feature_dims = feature_dims
        
        # 解码器阶段1：处理最深层特征
        self.decode_stage1 = nn.Sequential(
            nn.Conv2d(feature_dims[-1], feature_dims[-2], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            CBAM(feature_dims[-2]),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
        # 解码器阶段2：融合深层和中层特征
        self.fuse_stage2 = nn.Sequential(
            nn.Conv2d(feature_dims[-2] * 2, feature_dims[-2], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            CBAM(feature_dims[-2]),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(feature_dims[-2], feature_dims[-3], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            CBAM(feature_dims[-3])
        )
        
        # 解码器阶段3：融合中层和浅层特征
        self.fuse_stage3 = nn.Sequential(
            nn.Conv2d(feature_dims[-3] * 2, feature_dims[-3], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            CBAM(feature_dims[-3]),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(feature_dims[-3], feature_dims[-4], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            CBAM(feature_dims[-4])
        )
        
        # 解码器阶段4：融合浅层和原始特征
        self.fuse_stage4 = nn.Sequential(
            nn.Conv2d(feature_dims[-4] * 2, feature_dims[-4], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            CBAM(feature_dims[-4]),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(feature_dims[-4], feature_dims[-4] // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            CBAM(feature_dims[-4] // 2)
        )
        
        # 最终上采样和输出
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(feature_dims[-4] // 2, feature_dims[-4] // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            CBAM(feature_dims[-4] // 4),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(feature_dims[-4] // 4, out_channels, kernel_size=3, padding=1)
        )
        
        # 残差连接
        self.residual = nn.Identity()
    
    def forward(self, features):
        # 确保我们有足够的特征层
        assert len(features) >= 4, f"Expected at least 4 feature levels, got {len(features)}"
        
        # 从深到浅排序特征
        deep_feat = features[-1]  # 最深层
        mid_feat = features[-2]    # 中层
        shallow_feat = features[-3] # 浅层
        raw_feat = features[-4]     # 最浅层
        
        # 阶段1：处理最深层特征
        stage1_out = self.decode_stage1(deep_feat)
        
        # 阶段2：融合深层和中层特征
        # 确保尺寸匹配
        if stage1_out.shape != mid_feat.shape:
            mid_feat = nn.functional.interpolate(
                mid_feat, size=stage1_out.shape[2:], mode='bilinear', align_corners=True
            )
        stage2_in = torch.cat([stage1_out, mid_feat], dim=1)
        stage2_out = self.fuse_stage2(stage2_in)
        
        # 阶段3：融合中层和浅层特征
        if stage2_out.shape != shallow_feat.shape:
            shallow_feat = nn.functional.interpolate(
                shallow_feat, size=stage2_out.shape[2:], mode='bilinear', align_corners=True
            )
        stage3_in = torch.cat([stage2_out, shallow_feat], dim=1)
        stage3_out = self.fuse_stage3(stage3_in)
        
        # 阶段4：融合浅层和原始特征
        if stage3_out.shape != raw_feat.shape:
            raw_feat = nn.functional.interpolate(
                raw_feat, size=stage3_out.shape[2:], mode='bilinear', align_corners=True
            )
        stage4_in = torch.cat([stage3_out, raw_feat], dim=1)
        stage4_out = self.fuse_stage4(stage4_in)
        
        # 最终输出
        output = self.final_upsample(stage4_out)
        
        return output

class MAEPretrainer(nn.Module):
    """
    MAE (Masked Autoencoder) 预训练模型
    使用Swin Transformer作为编码器，带有km注意力机制
    """
    def __init__(self, backbone, mask_ratio=0.75, mask_patch_size=16):
        super().__init__()
        self.backbone = backbone
        self.mask_ratio = mask_ratio
        self.mask_patch_size = mask_patch_size
        
        # 定义特征维度，根据增大后的Swin Transformer设置
        # embed_dim=128, depths=(2, 4, 12, 4)
        # 输出通道数：128, 256, 512, 1024
        feature_dims = [128, 256, 512, 1024]
        
        # 创建多尺度特征融合解码器
        self.decoder = MultiScaleFusionDecoder(feature_dims, out_channels=2)
        
    def generate_multiscale_mask(self, x):
        """
        生成多尺度BEiT风格的掩码：混合使用不同大小的块进行掩码
        
        Args:
            x: 输入图像，形状为 (B, C, H, W)
        
        Returns:
            mask: 多尺度BEiT风格的掩码，形状为 (B, 1, H, W)
        """
        batch_size, _, height, width = x.shape
        
        # 定义多种掩码块大小
        mask_patch_sizes = torch.tensor([8, 16, 32], device=x.device)
        
        # 随机选择一种掩码块大小
        mask_patch_size = mask_patch_sizes[torch.randint(0, len(mask_patch_sizes), (1,), device=x.device)].item()
        
        # 计算掩码块的数量
        patch_height = height // mask_patch_size
        patch_width = width // mask_patch_size
        num_patches = patch_height * patch_width
        num_masked = int(num_patches * self.mask_ratio)
        
        # 生成掩码索引
        mask_indices = torch.rand(batch_size, num_patches, device=x.device).argsort(dim=-1) < num_masked
        mask_indices = mask_indices.view(batch_size, patch_height, patch_width)
        
        # 扩展掩码到像素级别
        mask = mask_indices.repeat_interleave(mask_patch_size, dim=1)
        mask = mask.repeat_interleave(mask_patch_size, dim=2)
        mask = mask.unsqueeze(1).float()
        
        # 确保掩码尺寸与输入匹配
        if mask.shape[-2:] != (height, width):
            mask = nn.functional.interpolate(
                mask, size=(height, width), mode='nearest'
            )
        
        return mask
    
    def forward(self, x):
        # 生成多尺度BEiT风格的掩码
        batch_size, _, height, width = x.shape
        mask = self.generate_multiscale_mask(x)
        masked_x = x * (1 - mask)  # BEiT使用1表示掩码区域
        
        # 通过骨干网络提取多尺度特征
        features = self.backbone(masked_x)
        
        # 确保我们有足够的特征
        if len(features) < 4:
            # 如果特征不够，复制深层特征
            while len(features) < 4:
                features.insert(0, features[0])
        
        # 解码重建图像
        reconstructed_x = self.decoder(features)
        
        # 确保输出尺寸与输入匹配
        if reconstructed_x.shape != x.shape:
            reconstructed_x = nn.functional.interpolate(
                reconstructed_x, size=(height, width), mode='bilinear', align_corners=True
            )
        
        return reconstructed_x, mask, features, x

def mae_pretrain(dataset_path='/hy-tmp/project/cell_segmentation/datasets/hdab_pannuke', fold=0):
    # 1. 加载H-DAB双通道数据集
    train_image_dir = os.path.join(dataset_path, f'fold{fold}', 'images')
    train_label_dir = os.path.join(dataset_path, f'fold{fold}', 'labels')
    
    print(f"Loading H-DAB dataset from {train_image_dir}")
    
    # 定义兼容image和mask关键字参数的transforms
    class CustomTransform:
        def __init__(self):
            # 导入必要的库
            import numpy as np
            from PIL import Image
            # 将导入的库保存为实例变量，以便在__call__方法中使用
            self.np = np
            self.Image = Image
            
            # 使用transforms.Compose组合所有变换
            # 注意：对于RGB转换后的图像，需要3通道的归一化参数
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # RGB三通道归一化
            ])
            # 对于mask，只需要调整大小和翻转，不需要归一化和颜色调整
            self.mask_transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor()
            ])
        
        def __call__(self, image=None, mask=None):
            # 确保image不为None
            if image is None:
                raise ValueError("image cannot be None")
                
            # 转换numpy数组为PIL图像（如果需要）
            if isinstance(image, self.np.ndarray):
                # H-DAB是双通道数据，需要特殊处理
                # 先将数据转换到0-255范围
                image = (image * 255).astype(self.np.uint8)
                # 对于双通道图像，我们需要将其转换为RGB格式以应用数据增强
                # 创建一个RGB图像，其中两个通道分别映射到R和G通道
                rgb_image = self.np.zeros((image.shape[0], image.shape[1], 3), dtype=self.np.uint8)
                rgb_image[:, :, 0] = image[:, :, 0]  # H通道到R通道
                rgb_image[:, :, 1] = image[:, :, 1]  # DAB通道到G通道
                image = self.Image.fromarray(rgb_image, mode='RGB')
            
            # 应用变换
            image = self.transform(image)
            
            # 只保留前两个通道（H和DAB）
            image = image[:2, :, :]
            
            # 如果有mask也处理mask
            if mask is not None:
                if isinstance(mask, self.np.ndarray):
                    # 处理3维mask数组，只取第一个通道
                    if mask.ndim == 3:
                        mask = mask[:, :, 0]  # 只取第一个通道
                    mask = self.Image.fromarray(mask, mode='L')
                mask = self.mask_transform(mask)
                return {'image': image, 'mask': mask}
            else:
                return {'image': image}
    
    train_dataset = PanNukeDataset(
        dataset_path=dataset_path,
        folds=fold,
        transforms=CustomTransform()
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    # 2. 初始化Swin Transformer模型作为编码器
    swin_backbone = SwinTransformerWithKMAttention(
        img_size=224,
        patch_size=4,
        in_chans=2,  # H-DAB双通道输入
        window_size=7,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        extract_layers=None  # 不指定提取层，使用默认设置
    )
    
    # 创建MAE预训练模型
    model = MAEPretrainer(backbone=swin_backbone, mask_ratio=0.75)
    
    # 4. 实现混合损失函数，包含知识蒸馏
    class HybridLoss(nn.Module):
        def __init__(self, alpha=1.0, beta=0.5, gamma=0.1, delta=0.2, device=None):
            super().__init__()
            self.alpha = alpha  # MSE损失权重
            self.beta = beta    # L1损失权重
            self.gamma = gamma  # 感知损失权重
            self.delta = delta  # 知识蒸馏损失权重
            self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 基础损失函数
            self.mse_loss = nn.MSELoss()
            self.l1_loss = nn.L1Loss()
            
            # 用于将双通道转换为三通道的卷积层
            self.channel_expander = nn.Conv2d(2, 3, kernel_size=1, bias=False)
            # 初始化权重，将两个通道映射到RGB通道
            nn.init.constant_(self.channel_expander.weight.data[:, 0, :, :], 1.0 / 2)
            nn.init.constant_(self.channel_expander.weight.data[:, 1, :, :], 1.0 / 2)
            # 移动到设备
            self.channel_expander.to(self.device)
            
            # 知识蒸馏：加载预训练的教师模型
            from cell_segmentation.models.backbone.swin_transformer import SwinTransformerWithKMAttention
            self.teacher_model = SwinTransformerWithKMAttention(
                in_chans=2, 
                extract_layers=[0, 1, 2, 3],
                layer_scale_init_value=1e-6
            )
            
            # 加载指定的预训练权重
            pretrained_path = "/hy-tmp/project/models/pretrained/CellViT-SAM-H-x40-004.pth"
            print(f"Loading pretrained weights from {pretrained_path} for teacher model")
            state_dict = torch.load(pretrained_path, map_location=self.device)
            
            # 只加载编码器部分的权重
            encoder_state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items() if 'backbone' in k}
            self.teacher_model.swin.load_state_dict(encoder_state_dict, strict=False)
            
            # 冻结教师模型权重
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            
            # 移动到设备
            self.teacher_model.to(self.device)
            
            # 感知损失特征提取器：使用简化的ResNet18（不下载权重）
            from torchvision.models import resnet18
            self.feature_extractor = resnet18(pretrained=False)
            # 冻结特征提取器的权重
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            # 只使用前几层
            self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:4])
            # 移动到设备
            self.feature_extractor.to(self.device)
        
        def forward(self, generated, target, mask=None, student_features=None, input_images=None):
            # 计算MSE损失
            mse_loss = self.mse_loss(generated, target)
            
            # 计算L1损失
            l1_loss = self.l1_loss(generated, target)
            
            # 计算感知损失
            # 将双通道转换为三通道
            generated_rgb = self.channel_expander(generated)
            target_rgb = self.channel_expander(target)
            
            # 提取特征
            generated_features = self.feature_extractor(generated_rgb)
            target_features = self.feature_extractor(target_rgb)
            
            # 计算特征损失
            perceptual_loss = self.mse_loss(generated_features, target_features)
            
            # 如果提供了掩码，只在掩码区域计算损失
            if mask is not None:
                # 计算掩码区域的像素数
                mask_area = torch.sum(1 - mask)
                if mask_area > 0:
                    # 在掩码区域计算损失
                    mse_loss = self.mse_loss(generated * (1 - mask), target * (1 - mask))
                    l1_loss = self.l1_loss(generated * (1 - mask), target * (1 - mask))
            
            # 计算知识蒸馏损失
            distillation_loss = 0.0
            if student_features is not None and input_images is not None:
                # 使用教师模型提取特征
                with torch.no_grad():
                    teacher_features = self.teacher_model(input_images)
                
                # 确保教师和学生特征数量匹配
                min_len = min(len(student_features), len(teacher_features))
                
                # 计算每一层的特征损失
                for i in range(min_len):
                    # 确保尺寸匹配
                    if student_features[i].shape != teacher_features[i].shape:
                        teacher_feat = nn.functional.interpolate(
                            teacher_features[i], 
                            size=student_features[i].shape[2:], 
                            mode='bilinear', 
                            align_corners=True
                        )
                    else:
                        teacher_feat = teacher_features[i]
                    
                    # 计算特征蒸馏损失
                    distillation_loss += self.mse_loss(student_features[i], teacher_feat)
                
                # 平均各层损失
                distillation_loss /= min_len
            
            # 组合损失
            total_loss = (
                self.alpha * mse_loss + 
                self.beta * l1_loss + 
                self.gamma * perceptual_loss + 
                self.delta * distillation_loss
            )
            
            return total_loss
    
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 3. 设置优化器和损失函数
    loss_fn = HybridLoss(alpha=1.0, beta=0.5, gamma=0.1, delta=0.2, device=device)
    
    # 使用带有梯度裁剪的AdamW优化器
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=1e-4, 
        weight_decay=0.05,
        betas=(0.9, 0.95)  # 使用更好的动量参数
    )
    
    # 使用带有预热的余弦退火学习率调度
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=20,  # 第一个周期的迭代次数
        T_mult=2,  # 后续周期迭代次数的倍数
        eta_min=1e-6  # 最小学习率
    )
    
    # 创建保存目录
    os.makedirs('checkpoints/mae', exist_ok=True)
    
    # 4. 训练循环
    num_epochs = 100
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_data in progress_bar:
            # PanNukeDataset返回4个值：img, masks, tissue_type, img_name
            if len(batch_data) == 4:
                images, masks_dict, tissue_type, img_name = batch_data
            else:
                # 兼容不同的返回值格式
                images = batch_data[0]
            
            # 确保图像是双通道的
            if images.size(1) != 2:
                # 如果不是双通道，尝试选择前两个通道或重复通道
                if images.size(1) > 2:
                    images = images[:, :2]
                elif images.size(1) == 1:
                    images = images.repeat(1, 2, 1, 1)
            
            images = images.to(device)
            
            # 前向传播
            reconstructed_images, masks, student_features, input_images = model(images)
            
            # 计算损失（包含知识蒸馏损失）
            loss = loss_fn(
                reconstructed_images, 
                images, 
                masks, 
                student_features, 
                input_images
            )
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item() * images.size(0)
            progress_bar.set_postfix({'loss': loss.item()})
        
        # 更新学习率
        scheduler.step()
        
        # 计算平均损失
        avg_loss = epoch_loss / len(train_dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.6f}')
        
        # 保存模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = f'checkpoints/mae/mae_fold{fold}_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, save_path)
            print(f'Best model saved to {save_path}')
    
    # 训练结束，保存最后一个epoch的模型
    final_save_path = f'checkpoints/mae/mae_fold{fold}_final.pth'
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, final_save_path)
    print(f'Final model saved to {final_save_path}')
    
    # 只保存编码器权重用于后续微调
    encoder_save_path = f'checkpoints/mae/mae_encoder_fold{fold}.pth'
    torch.save(model.backbone.state_dict(), encoder_save_path)
    print(f'Encoder weights saved to {encoder_save_path}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MAE Pretraining for Cell Segmentation')
    parser.add_argument('--dataset_path', type=str, default='/hy-tmp/project/cell_segmentation/datasets/hdab_pannuke', help='Path to the H-DAB dataset')
    parser.add_argument('--fold', type=int, default=0, help='Fold number for training')
    args = parser.parse_args()
    
    mae_pretrain(args.dataset_path, args.fold)
    
    # 确保保存目录存在
    os.makedirs('checkpoints', exist_ok=True)
    
    # 4. 训练100 epochs
    for epoch in range(100):
        model.train()
        total_loss = 0
        
        for images, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/100"):
            # 确保图像是双通道的
            if images.size(1) != 2:
                # 如果不是双通道，尝试选择前两个通道
                images = images[:, :2]
            
            images = images.to(device)
            
            # 前向传播
            reconstructed_images, mask = model(images)
            
            # 计算损失
            loss = loss_fn(reconstructed_images, images)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Pretrain Epoch {epoch+1}/100, Loss: {avg_loss:.4f}")
        
        # 学习率调度
        scheduler.step()
        
        # 保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'checkpoints/mae_pretrained_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
    
    # 保存最终模型
    torch.save(model.state_dict(), 'mae_pretrained_final.pth')
    print("MAE pre-training completed!")
    
    # 提取并保存骨干网络权重（用于下游任务）
    torch.save(model.backbone.state_dict(), 'swin_backbone_pretrained.pth')
    print("Backbone weights saved for downstream tasks!")

