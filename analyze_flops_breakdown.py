
import torch
import torch.nn as nn
from thop import profile, clever_format
from models.segmentation.cell_segmentation.sparse_cellvit import SparseCellViT

def analyze_flops_breakdown():
    print("=" * 60)
    print("FLOPs Breakdown Analysis")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model Config
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
    
    model = SparseCellViT(**model_config).to(device)
    model.eval()
    
    input_size = (1, 3, 256, 256)
    dummy_input = torch.randn(input_size).to(device)
    
    print(f"Input Size: {input_size}")
    
    # 1. Total FLOPs
    total_macs, _ = profile(model, inputs=(dummy_input, ), verbose=False)
    total_flops = total_macs * 2
    print(f"Total FLOPs: {total_flops / 1e9:.3f} G")
    
    # 2. Backbone FLOPs
    # We need to hook the backbone forward or run it separately
    # Let's run backbone separately
    backbone_input = dummy_input
    backbone_macs, _ = profile(model.backbone, inputs=(backbone_input, ), verbose=False)
    backbone_flops = backbone_macs * 2
    print(f"Backbone FLOPs: {backbone_flops / 1e9:.3f} G ({backbone_flops/total_flops:.1%})")
    
    # 3. Decoder Heads FLOPs (Approximation)
    # We can't easily isolate decoders because they depend on intermediate features.
    # But we can inspect the `decoder0_header` of one branch if we construct inputs for it.
    
    # Let's construct a dummy input for decoder0_header
    # The input to decoder0_header is concatenation of z0 (input) and b1 (upsampled features)
    # z0: [1, 3, 256, 256]
    # b1: [1, 64, 256, 256] (from decoder1_upsampler, assuming channel match)
    # Actually, let's look at the definition in sparse_cellvit.py
    # decoder0_header input channels: self.input_channels + 64 = 67
    
    header_input = torch.randn(1, 67, 256, 256).to(device)
    
    # Binary Decoder Header
    binary_header = model.nuclei_binary_map_decoder.decoder0_header
    bin_head_macs, _ = profile(binary_header, inputs=(header_input, ), verbose=False)
    bin_head_flops = bin_head_macs * 2
    print(f"Binary Decoder Header (Final Stage) FLOPs: {bin_head_flops / 1e9:.3f} G")
    
    # HV Decoder Header
    hv_header = model.hv_map_decoder.decoder0_header
    hv_head_macs, _ = profile(hv_header, inputs=(header_input, ), verbose=False)
    hv_head_flops = hv_head_macs * 2
    print(f"HV Decoder Header (Final Stage) FLOPs: {hv_head_flops / 1e9:.3f} G")
    
    # Type Decoder Header
    type_header = model.nuclei_type_maps_decoder.decoder0_header
    type_head_macs, _ = profile(type_header, inputs=(header_input, ), verbose=False)
    type_head_flops = type_head_macs * 2
    print(f"Type Decoder Header (Final Stage) FLOPs: {type_head_flops / 1e9:.3f} G")
    
    # Sum of Headers
    headers_sum = bin_head_flops + hv_head_flops + type_head_flops
    print(f"Sum of 3 Headers (Final Stage Only): {headers_sum / 1e9:.3f} G ({headers_sum/total_flops:.1%})")
    
    # 4. Upsampling Path (Rest of Decoders)
    # Total - Backbone - Headers = Upsampling Path + Tissue Head
    upsampling_flops = total_flops - backbone_flops - headers_sum
    print(f"Upsampling Path & Others FLOPs: {upsampling_flops / 1e9:.3f} G ({upsampling_flops/total_flops:.1%})")

if __name__ == "__main__":
    analyze_flops_breakdown()
