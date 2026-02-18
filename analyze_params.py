
import torch
import torch.nn as nn
from models.segmentation.cell_segmentation.sparse_cellvit import SparseCellViT
try:
    from torchinfo import summary
except ImportError:
    print("Installing torchinfo...")
    import subprocess
    subprocess.check_call(["pip", "install", "torchinfo"])
    from torchinfo import summary

def analyze_params():
    print("=" * 60)
    print("Deep Parameter Analysis")
    print("=" * 60)
    
    # 1. Initialize Model
    # Exact configuration from train_from_scratch.py
    model_config = {
        "num_nuclei_classes": 6,
        "num_tissue_classes": 19,
        "embed_dim": 128,
        "input_channels": 3,
        "depth": 16,
        "num_heads": 4,
        "extract_layers": [4, 8, 12, 16],
        "regression_loss": True,
        "window_size": 16,
        "k": 0.2,
        "dynamic_k": True,
        "min_k_ratio": 0.1,
        "max_k_ratio": 0.7,
        "sparsity_level": 0.2,
        "mlp_ratio": 3.0,
        "drop_rate": 0.1,
        "attn_drop_rate": 0.1,
        "drop_path_rate": 0.1
    }
    
    print("Model Configuration:")
    for k, v in model_config.items():
        print(f"  {k}: {v}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SparseCellViT(**model_config).to(device)
    model.eval()
    
    # 2. Detailed Breakdown using torchinfo
    print("\n[Layer-wise Parameter Breakdown]")
    input_size = (1, 3, 256, 256)
    
    # Use torchinfo for pretty printing
    model_stats = summary(model, input_size=input_size, verbose=0, 
                          col_names=["input_size", "output_size", "num_params", "mult_adds"],
                          row_settings=["var_names", "depth"])
    print(model_stats)
    
    # 3. Manual Breakdown by Component
    print("\n[Component-wise Breakdown]")
    
    def count_params(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    backbone_params = count_params(model.backbone)
    tissue_head_params = count_params(model.tissue_head)
    
    # Decoders
    decoder_params = 0
    decoder_components = [
        ("bottleneck_upsampler", model.nuclei_binary_map_decoder[0]), # Accessing through Sequential might be tricky, let's use named_modules approach partially
    ]
    
    # Since decoders are shared in structure but instantiated separately via create_upsampling_branch
    # Wait, in SparseCellViT, self.nuclei_binary_map_decoder is a Sequential created by create_upsampling_branch
    
    binary_decoder_params = count_params(model.nuclei_binary_map_decoder)
    hv_decoder_params = count_params(model.hv_map_decoder)
    type_decoder_params = count_params(model.nuclei_type_maps_decoder)
    
    total_params = count_params(model)
    
    print(f"Backbone (Swin-T): {backbone_params:,} ({backbone_params/total_params:.1%})")
    print(f"Tissue Head: {tissue_head_params:,} ({tissue_head_params/total_params:.1%})")
    print(f"Nuclei Binary Decoder: {binary_decoder_params:,} ({binary_decoder_params/total_params:.1%})")
    print(f"HV Map Decoder: {hv_decoder_params:,} ({hv_decoder_params/total_params:.1%})")
    print(f"Nuclei Type Decoder: {type_decoder_params:,} ({type_decoder_params/total_params:.1%})")
    
    print(f"\nTotal Trainable Parameters: {total_params:,}")
    
    # Check if calculation matches
    sum_components = backbone_params + tissue_head_params + binary_decoder_params + hv_decoder_params + type_decoder_params
    print(f"Sum of main components: {sum_components:,}")
    print(f"Difference (shared/misc): {total_params - sum_components:,}")

if __name__ == "__main__":
    analyze_params()
