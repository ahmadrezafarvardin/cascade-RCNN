"""
YOLO inference script for character detection
"""
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.yolo import YOLOCharacterDetector
from ultralytics import YOLO
import json
import cv2

def load_trained_model(checkpoint='best'):
    """Load trained YOLO model
    
    Args:
        checkpoint: 'best', 'last', or path to specific weights
    """
    if checkpoint in ['best', 'last']:
        weights_path = Path(f'results/yolo_runs/detect/character_detection/weights/{checkpoint}.pt')
        if not weights_path.exists():
            print(f"No {checkpoint} weights found. Using pretrained model.")
            return YOLOCharacterDetector('yolov8n.pt')
    else:
        weights_path = Path(checkpoint)
    
    if weights_path.exists():
        print(f"Loading weights from: {weights_path}")
        detector = YOLOCharacterDetector()
        detector.model = YOLO(str(weights_path))
        return detector
    else:
        raise FileNotFoundError(f"Weights not found: {weights_path}")

def run_inference(image_path, model=None, save_viz=True, conf_threshold=0.25):
    """Run inference on a single image
    
    Args:
        image_path: Path to image
        model: YOLO model (if None, will load best weights)
        save_viz: Whether to save visualization
        conf_threshold: Confidence threshold for detections
    """
    if model is None:
        model = load_trained_model('best')
    
    # Run inference
    results = model.predict(
        image_path, 
        conf=conf_threshold,
        save=save_viz,
        project='results/yolo_predictions' if save_viz else None
    )
    
    # Extract predictions
    predictions = []
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                predictions.append({
                    'bbox': [x1, y1, x2-x1, y2-y1],  # Convert to xywh
                    'confidence': float(box.conf[0]),
                    'class': 'char'
                })
    
    return predictions

def batch_inference(image_dir, output_json='results/yolo_predictions.json', model=None):
    """Run inference on multiple images
    
    Args:
        image_dir: Directory containing images
        output_json: Path to save predictions
        model: YOLO model (if None, will load best weights)
    """
    if model is None:
        model = load_trained_model('best')
    
    image_dir = Path(image_dir)
    all_predictions = {}
    
    # Process all images
    image_files = list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg'))
    print(f"Processing {len(image_files)} images...")
    
    for img_path in image_files:
        predictions = run_inference(str(img_path), model=model, save_viz=False)
        all_predictions[img_path.name] = {
            'predictions': predictions,
            'num_chars': len(predictions)
        }
        print(f"  {img_path.name}: {len(predictions)} characters detected")
    
    # Save results
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(all_predictions, f, indent=2)
    
    print(f"\nPredictions saved to: {output_json}")
    return all_predictions

def visualize_predictions(image_path, predictions=None, model=None):
    """Visualize predictions on an image"""
    if predictions is None:
        predictions = run_inference(image_path, model=model, save_viz=False)
    
    # Load image
    img = cv2.imread(str(image_path))
    
    # Draw boxes
    for pred in predictions:
        x, y, w, h = pred['bbox']
        conf = pred['confidence']
        
        # Draw rectangle
        cv2.rectangle(img, 
                     (int(x), int(y)), 
                     (int(x+w), int(y+h)), 
                     (0, 255, 0), 2)
        
        # Add confidence score
        cv2.putText(img, f'{conf:.2f}', 
                   (int(x), int(y-5)),
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (0, 255, 0), 1)
    
    # Save visualization
    output_path = f'results/yolo_predictions/viz_{Path(image_path).name}'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"Visualization saved to: {output_path}")
    
    return img

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='YOLO inference for character detection')
    parser.add_argument('input', help='Image path or directory')
    parser.add_argument('--weights', default='best', 
                        help='Weights to use: best, last, or path to .pt file')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--visualize', action='store_true',
                        help='Save visualizations')
    
    args = parser.parse_args()
    
    # Check if input is file or directory
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image inference
        predictions = run_inference(
            str(input_path), 
            save_viz=args.visualize,
            conf_threshold=args.conf
        )
        print(f"\nDetected {len(predictions)} characters")
        for i, pred in enumerate(predictions[:5]):  # Show first 5
            print(f"  Char {i+1}: bbox={pred['bbox']}, conf={pred['confidence']:.3f}")
            
    elif input_path.is_dir():
        # Batch inference
        batch_inference(
            str(input_path),
            output_json=f'results/yolo_predictions_{input_path.name}.json'
        )
    else:
        print(f"Error: {input_path} not found")
