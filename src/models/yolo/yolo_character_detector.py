"""
Simple YOLO wrapper for character detection
"""
from ultralytics import YOLO
import os
from pathlib import Path

class YOLOCharacterDetector:
    def __init__(self, model_name='yolov8n.pt'):
        # Model weights are in the same directory as this file
        model_path = Path(__file__).parent / model_name
        if not model_path.exists():
            # Download if not exists
            self.model = YOLO(model_name)
        else:
            self.model = YOLO(str(model_path))
    
    def train(self, data_yaml='dataset_yolo/data.yaml', **kwargs):
        """Train the model with default settings for character detection"""
        default_args = {
            'epochs': 100,
            'imgsz': 640,
            'batch': 8,
            'device': 0,
            'workers': 4,
            'amp': False,
            'project': 'results/yolo_runs',
            'name': 'character_detection',
            'exist_ok': True,
            'patience': 20,
            'save': True,
            'save_period': 10,
            'val': True,
            'plots': True,
            'cache': False,
            'rect': False,
            'mosaic': 0.5,
            'mixup': 0.0,
            'copy_paste': 0.0,
        }
        # Override with any provided arguments
        default_args.update(kwargs)
        
        return self.model.train(data=data_yaml, **default_args)
    
    def predict(self, source, **kwargs):
        """Run inference"""
        return self.model(source, **kwargs)
    
    def val(self, **kwargs):
        """Validate the model"""
        return self.model.val(**kwargs)
