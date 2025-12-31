#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv8 验证示例脚本

使用方法:
    python val_example.py --model runs/detect/train/weights/best.pt
    python val_example.py --model runs/detect/train/weights/best.pt --data coco128.yaml
"""

from ultralytics import YOLO
import argparse


def validate_model(model_path, data=None, imgsz=640, batch=16, 
                   conf=0.001, iou=0.7, device=0, plots=True):
    """
    验证 YOLOv8 模型
    
    参数:
        model_path: 模型文件路径
        data: 数据集配置文件（可选，模型会记住训练时的配置）
        imgsz: 图像尺寸
        batch: 批次大小
        conf: 置信度阈值
        iou: IoU 阈值
        device: 设备
        plots: 是否生成图表
    """
    print(f"🔍 开始验证模型")
    print(f"🤖 模型: {model_path}")
    print(f"📦 数据集: {data or '使用模型保存的配置'}")
    print(f"📐 图像尺寸: {imgsz}")
    print(f"📊 批次大小: {batch}")
    print(f"🎯 置信度阈值: {conf}")
    print(f"📏 IoU 阈值: {iou}")
    print("-" * 50)
    
    # 加载模型
    print(f"📥 加载模型...")
    model = YOLO(model_path)
    
    # 验证
    print(f"🔍 开始验证...")
    kwargs = {
        'imgsz': imgsz,
        'batch': batch,
        'conf': conf,
        'iou': iou,
        'device': device,
        'plots': plots,
        'verbose': True
    }
    
    if data:
        kwargs['data'] = data
    
    results = model.val(**kwargs)
    
    print("-" * 50)
    print("✅ 验证完成！")
    
    # 显示关键指标
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print(f"\n📊 验证指标:")
        print(f"   mAP50: {metrics.get('metrics/mAP50(B)', 'N/A'):.4f}")
        print(f"   mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
        print(f"   精度 (P): {metrics.get('metrics/precision(B)', 'N/A'):.4f}")
        print(f"   召回率 (R): {metrics.get('metrics/recall(B)', 'N/A'):.4f}")
    else:
        # 尝试从验证器获取指标
        if hasattr(model, 'validator') and model.validator:
            validator = model.validator
            print(f"\n📊 验证指标:")
            if hasattr(validator, 'metrics'):
                print(f"   精度: {validator.metrics.get('precision', 'N/A'):.4f}")
                print(f"   召回率: {validator.metrics.get('recall', 'N/A'):.4f}")
                print(f"   mAP50: {validator.metrics.get('map50', 'N/A'):.4f}")
                print(f"   mAP50-95: {validator.metrics.get('map', 'N/A'):.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 验证脚本")
    parser.add_argument("--model", type=str, required=True,
                       help="模型文件路径（.pt）")
    parser.add_argument("--data", type=str, default=None,
                       help="数据集配置文件路径（可选）")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="图像尺寸")
    parser.add_argument("--batch", type=int, default=16,
                       help="批次大小")
    parser.add_argument("--conf", type=float, default=0.001,
                       help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.7,
                       help="IoU 阈值")
    parser.add_argument("--device", type=str, default="0",
                       help="设备（0 表示 GPU 0，'cpu' 表示 CPU）")
    parser.add_argument("--plots", action="store_true", default=True,
                       help="生成图表")
    
    args = parser.parse_args()
    
    # 转换 device 参数
    if args.device == "cpu":
        device = "cpu"
    elif args.device.isdigit():
        device = int(args.device)
    else:
        device = args.device
    
    # 验证模型
    validate_model(
        model_path=args.model,
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=device,
        plots=args.plots
    )


if __name__ == "__main__":
    main()

