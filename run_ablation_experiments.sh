#!/bin/bash
# SparseSwinCell 消融实验批量运行脚本

echo "=========================================="
echo "SparseSwinCell 消融实验开始"
echo "=========================================="

# 定义要运行的消融实验（已有完整模型，只运行消融实验）
ABLATIONS=("no_sparse_attention" "no_shape_stream" "no_aspp" "no_attention_gates")

# 记录开始时间
START_TIME=$(date)
echo "开始时间: $START_TIME"
echo ""

# 逐个运行消融实验
for ablation in "${ABLATIONS[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "运行消融实验: $ablation"
    echo "----------------------------------------"
    echo ""
    
    python cell_segmentation/trainer/ablation_study.py --ablation "$ablation"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ 消融实验 $ablation 完成成功"
    else
        echo ""
        echo "✗ 消融实验 $ablation 失败"
        echo "继续运行下一个实验..."
    fi
    
    echo ""
    echo "等待 5 秒后继续..."
    sleep 5
done

echo ""
echo "=========================================="
echo "所有消融实验完成！"
echo "=========================================="

# 记录结束时间
END_TIME=$(date)
echo "结束时间: $END_TIME"
echo ""
echo "开始分析结果..."
echo ""

# 运行结果分析
python cell_segmentation/trainer/analyze_ablation_results.py

echo ""
echo "=========================================="
echo "消融实验和分析全部完成！"
echo "=========================================="

