#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv8 训练示例脚本

使用方法:
    python train_example.py

或者修改参数后运行:
    python train_example.py --data my_dataset.yaml --epochs 100
"""

from ultralytics import YOLO
import argparse


def train_yolo(data="coco128.yaml", model="yolov8n.pt", epochs=100, imgsz=640, 
               batch=16, device=0, project="runs", name="train"):
    """
    训练 YOLOv8 模型
    
    参数:
        data: 数据集配置文件路径
        model: 模型文件路径（.pt 或 .yaml）
        epochs: 训练轮数
        imgsz: 图像尺寸
        batch: 批次大小
        device: 设备（0 表示 GPU 0，'cpu' 表示 CPU）
        project: 项目名称
        name: 实验名称
    """
    print(f"🚀 开始训练 YOLOv8 模型")
    print(f"📦 数据集: {data}")
    print(f"🤖 模型: {model}")
    print(f"🔄 训练轮数: {epochs}")
    print(f"📐 图像尺寸: {imgsz}")
    print(f"📊 批次大小: {batch}")
    print(f"💻 设备: {device}")
    print("-" * 50)
    
    # 加载模型
    print(f"📥 加载模型: {model}")
    model = YOLO(model)
    
    # 开始训练
    print(f"🎯 开始训练...")
    results = model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        # 训练参数
        patience=50,  # 早停耐心值
        save=True,    # 保存检查点
        plots=True,   # 生成图表
        verbose=True, # 详细输出
    )
    
    print("-" * 50)
    print("✅ 训练完成！")
    print(f"📁 结果保存在: {results.save_dir}")
    print(f"🏆 最佳模型: {results.save_dir}/weights/best.pt")
    print(f"💾 最后模型: {results.save_dir}/weights/last.pt")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 训练脚本")
    parser.add_argument("--data", type=str, default="coco128.yaml", 
                       help="数据集配置文件路径")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                       help="模型文件路径（.pt 或 .yaml）")
    parser.add_argument("--epochs", type=int, default=100,
                       help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="图像尺寸")
    parser.add_argument("--batch", type=int, default=16,
                       help="批次大小")
    parser.add_argument("--device", type=str, default="0",
                       help="设备（0 表示 GPU 0，'cpu' 表示 CPU）")
    parser.add_argument("--project", type=str, default="runs",
                       help="项目名称")
    parser.add_argument("--name", type=str, default="train",
                       help="实验名称")
    
    args = parser.parse_args()
    
    # 转换 device 参数
    if args.device == "cpu":
        device = "cpu"
    elif args.device.isdigit():
        device = int(args.device)
    else:
        device = args.device
    
    # 开始训练
    train_yolo(
        data=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name
    )


if __name__ == "__main__":
    main()

