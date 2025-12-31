#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv8 实时物体检测示例脚本

使用方法:
    # 检测图像
    python detect_example.py --model runs/train/weights/best.pt --source image.jpg
    
    # 检测视频
    python detect_example.py --model runs/train/weights/best.pt --source video.mp4
    
    # 摄像头实时检测
    python detect_example.py --model runs/train/weights/best.pt --source 0
    
    # 检测文件夹
    python detect_example.py --model runs/train/weights/best.pt --source images/
"""

from ultralytics import YOLO
import argparse
import os


def detect_objects(model_path, source, conf=0.25, iou=0.7, imgsz=640, 
                  device=0, save=True, show=False, save_txt=False, 
                  save_conf=False, save_crop=False, classes=None):
    """
    使用 YOLOv8 进行物体检测
    
    参数:
        model_path: 模型文件路径
        source: 输入源（图像/视频/摄像头/文件夹）
        conf: 置信度阈值
        iou: IoU 阈值
        imgsz: 图像尺寸
        device: 设备
        save: 是否保存结果
        show: 是否显示结果
        save_txt: 是否保存标签文件
        save_conf: 是否保存置信度
        save_crop: 是否保存裁剪的检测框
        classes: 只检测指定类别（列表）
    """
    print(f"🎯 开始物体检测")
    print(f"🤖 模型: {model_path}")
    print(f"📥 输入源: {source}")
    print(f"🎯 置信度阈值: {conf}")
    print(f"📏 IoU 阈值: {iou}")
    print(f"📐 图像尺寸: {imgsz}")
    print(f"💻 设备: {device}")
    print("-" * 50)
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"❌ 错误: 模型文件不存在: {model_path}")
        return None
    
    # 加载模型
    print(f"📥 加载模型...")
    model = YOLO(model_path)
    
    # 判断输入源类型
    source_type = "未知"
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        source_type = "摄像头"
        source = int(source) if isinstance(source, str) else source
    elif isinstance(source, str):
        if os.path.isfile(source):
            if source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                source_type = "视频"
            else:
                source_type = "图像"
        elif os.path.isdir(source):
            source_type = "文件夹"
        elif source.startswith('http'):
            source_type = "URL"
    
    print(f"📋 输入类型: {source_type}")
    print(f"🔍 开始检测...")
    print("-" * 50)
    
    # 检测参数
    kwargs = {
        'conf': conf,
        'iou': iou,
        'imgsz': imgsz,
        'device': device,
        'save': save,
        'show': show,
        'save_txt': save_txt,
        'save_conf': save_conf,
        'save_crop': save_crop,
        'verbose': True
    }
    
    # 如果是摄像头或视频，使用流式处理
    if source_type in ["摄像头", "视频"]:
        kwargs['stream'] = True
    
    # 指定检测类别
    if classes is not None:
        kwargs['classes'] = classes
    
    # 执行检测
    try:
        results = model.predict(source=source, **kwargs)
        
        # 处理流式结果
        if source_type in ["摄像头", "视频"]:
            detection_count = 0
            frame_count = 0
            
            for result in results:
                frame_count += 1
                num_detections = len(result.boxes)
                detection_count += num_detections
                
                if frame_count % 30 == 0:  # 每30帧打印一次
                    print(f"已处理 {frame_count} 帧, 检测到 {detection_count} 个目标")
            
            print("-" * 50)
            print(f"✅ 检测完成！")
            print(f"📊 总帧数: {frame_count}")
            print(f"🎯 总检测数: {detection_count}")
        else:
            # 处理图像或文件夹
            total_detections = 0
            for result in results:
                num_detections = len(result.boxes)
                total_detections += num_detections
                
                # 显示检测信息
                if num_detections > 0:
                    print(f"\n检测到 {num_detections} 个目标:")
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        confidence = float(box.conf[0])
                        print(f"  - {class_name}: {confidence:.2f}")
            
            print("-" * 50)
            print(f"✅ 检测完成！")
            print(f"🎯 总检测数: {total_detections}")
            
            # 显示结果保存位置
            if save and len(results) > 0:
                save_dir = results[0].save_dir if hasattr(results[0], 'save_dir') else "runs/detect/predict"
                print(f"📁 结果保存在: {save_dir}")
        
        return results
        
    except Exception as e:
        print(f"❌ 检测过程中出错: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 物体检测脚本")
    parser.add_argument("--model", type=str, default="runs/train/weights/best.pt",
                       help="模型文件路径（.pt）")
    parser.add_argument("--source", type=str, required=True,
                       help="输入源（图像/视频/摄像头/文件夹）。摄像头使用 0")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="置信度阈值 (0-1)")
    parser.add_argument("--iou", type=float, default=0.7,
                       help="IoU 阈值 (0-1)")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="图像尺寸")
    parser.add_argument("--device", type=str, default="0",
                       help="设备（0 表示 GPU 0，'cpu' 表示 CPU）")
    parser.add_argument("--save", action="store_true", default=True,
                       help="保存检测结果")
    parser.add_argument("--show", action="store_true", default=False,
                       help="显示检测结果（图像/视频）")
    parser.add_argument("--save-txt", action="store_true", default=False,
                       help="保存标签文件")
    parser.add_argument("--save-conf", action="store_true", default=False,
                       help="保存置信度")
    parser.add_argument("--save-crop", action="store_true", default=False,
                       help="保存裁剪的检测框")
    parser.add_argument("--classes", type=str, default=None,
                       help="只检测指定类别，用逗号分隔，如 '0,1,2'")
    
    args = parser.parse_args()
    
    # 转换 device 参数
    if args.device == "cpu":
        device = "cpu"
    elif args.device.isdigit():
        device = int(args.device)
    else:
        device = args.device
    
    # 转换 source 参数（如果是摄像头）
    source = args.source
    if source.isdigit():
        source = int(source)
    
    # 解析类别参数
    classes = None
    if args.classes:
        classes = [int(c.strip()) for c in args.classes.split(',')]
    
    # 执行检测
    detect_objects(
        model_path=args.model,
        source=source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=device,
        save=args.save,
        show=args.show,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        save_crop=args.save_crop,
        classes=classes
    )


if __name__ == "__main__":
    main()

