"""
YOLO training script for character detection
"""
import sys
import os
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.yolo import YOLOCharacterDetector
import json
import shutil
from PIL import Image

def convert_to_yolo_format():
    """Convert dataset to YOLO format with proper normalization"""
    for split in ['train', 'val']:
        # Create directories
        os.makedirs(f'dataset_yolo/{split}/images', exist_ok=True)
        os.makedirs(f'dataset_yolo/{split}/labels', exist_ok=True)
        
        img_dir = f'dataset/{split}/images'
        label_dir = f'dataset/{split}/labels'
        
        skipped_images = []
        
        for img_name in os.listdir(img_dir):
            if not img_name.endswith('.png'):
                continue
            
            # Load image to get dimensions
            img_path = os.path.join(img_dir, img_name)
            img = Image.open(img_path)
            img_w, img_h = img.size
            
            # Copy image
            shutil.copy(img_path, f'dataset_yolo/{split}/images/{img_name}')
            
            # Convert labels
            label_path = os.path.join(label_dir, img_name.replace('.png', '.json'))
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    label_data = json.load(f)
                
                # Convert to YOLO format with validation
                yolo_labels = []
                valid_image = True
                
                for ann in label_data.get('annotations', []):
                    bbox = ann['boundingBox']
                    
                    # Calculate YOLO format coordinates
                    x_min = bbox['x']
                    y_min = bbox['y']
                    x_max = x_min + bbox['width']
                    y_max = y_min + bbox['height']
                    
                    # Clip coordinates to image boundaries
                    x_min = max(0, min(x_min, img_w))
                    y_min = max(0, min(y_min, img_h))
                    x_max = max(0, min(x_max, img_w))
                    y_max = max(0, min(y_max, img_h))
                    
                    # Skip if box is invalid after clipping
                    if x_max <= x_min or y_max <= y_min:
                        continue
                    
                    # Convert to YOLO format (normalized center coordinates)
                    x_center = (x_min + x_max) / 2 / img_w
                    y_center = (y_min + y_max) / 2 / img_h
                    width = (x_max - x_min) / img_w
                    height = (y_max - y_min) / img_h
                    
                    # Validate normalized coordinates
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                            0 < width <= 1 and 0 < height <= 1):
                        valid_image = False
                        break
                    
                    yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                
                if valid_image and yolo_labels:
                    # Save YOLO labels
                    with open(f'dataset_yolo/{split}/labels/{img_name.replace(".png", ".txt")}', 'w') as f:
                        f.write('\n'.join(yolo_labels))
                else:
                    # Remove the copied image if labels are invalid
                    os.remove(f'dataset_yolo/{split}/images/{img_name}')
                    skipped_images.append(img_name)
        
        print(f"{split} set: Processed {len(os.listdir(f'dataset_yolo/{split}/images'))} images")

def create_data_yaml():
    """Create YOLO data.yaml file"""
    data_yaml = """path: dataset_yolo
train: train/images
val: val/images

nc: 1
names: ['character']
"""
    os.makedirs('dataset_yolo', exist_ok=True)
    with open('dataset_yolo/data.yaml', 'w') as f:
        f.write(data_yaml)

def train_yolo(model_size='n', resume=False):
    """Train YOLO model
    
    Args:
        model_size: 'n' for nano, 'm' for medium
        resume: Whether to resume from checkpoint
    """
    # Setup
    if not os.path.exists('dataset_yolo/data.yaml'):
        print("Preparing dataset...")
        convert_to_yolo_format()
        create_data_yaml()
    
    # Initialize model
    print(f"Initializing YOLOv8{model_size} model...")
    detector = YOLOCharacterDetector(f'yolov8{model_size}.pt')
    
    # Check for existing checkpoint
    checkpoint_path = Path('results/yolo_runs/detect/character_detection/weights/last.pt')
    if resume and checkpoint_path.exists():
        print(f"Resuming from checkpoint: {checkpoint_path}")
        from ultralytics import YOLO
        detector.model = YOLO(str(checkpoint_path))
        results = detector.model.train(resume=True)
    else:
        # Train from scratch or pretrained
        results = detector.train(
            data='dataset_yolo/data.yaml',
            epochs=100,
            imgsz=640,
            batch=8,
            name='character_detection'
        )
    
    print("\nTraining complete!")
    print(f"Results saved to: results/yolo_runs/detect/character_detection/")
    
    # Quick validation
    print("\nRunning validation...")
    metrics = detector.val()
    if hasattr(metrics, 'box'):
        print(f"mAP50: {metrics.box.map50:.3f}")
        print(f"mAP50-95: {metrics.box.map:.3f}")
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train YOLO for character detection')
    parser.add_argument('--model', choices=['n', 'm'], default='n', 
                        help='Model size: n (nano) or m (medium)')
    parser.add_argument('--resume', action='store_true', 
                        help='Resume from last checkpoint')
    
    args = parser.parse_args()
    
    train_yolo(model_size=args.model, resume=args.resume)
