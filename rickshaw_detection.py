import os
import cv2
import numpy as np
import shutil
import yaml
import random
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLO

class RickshawDetector:
    def __init__(self, data_dir='/Users/dibbyoroy/Desktop/ML project/rickshaw_detection/Rickshaw-Detection'):
        self.data_dir = data_dir
        self.labeled_images_dir = os.path.join(data_dir, 'rickshaw_labeled_images')
        self.yolo_dir = os.path.join(data_dir, 'yolo_dataset')
        self.classes = ['auto', 'non-auto']
        self.model = None
        
        os.makedirs(os.path.join(self.yolo_dir, 'images', 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.yolo_dir, 'images', 'val'), exist_ok=True)
        os.makedirs(os.path.join(self.yolo_dir, 'labels', 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.yolo_dir, 'labels', 'val'), exist_ok=True)
    
    def prepare_dataset(self, train_ratio=0.9):
        print("Preparing dataset for YOLO training...")
        
        # Process the labeled images
        self._process_labeled_images(train_ratio=train_ratio)
        
        # Validate the dataset
        self._validate_dataset()
        
        # Create YAML config file for training
        self._create_yaml_config()
        
        print("Dataset preparation completed!")
    
    def _process_labeled_images(self, train_ratio=0.9):
        """
        Process the labeled images from rickshaw_labeled_images folder
        """
        # Get list of all image files
        source_images_dir = os.path.join(self.labeled_images_dir, 'images')
        source_labels_dir = os.path.join(self.labeled_images_dir, 'labels')
        
        # Get all image files
        image_files = [f for f in os.listdir(source_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Shuffle and split into train/val sets
        random.shuffle(image_files)
        split_idx = int(len(image_files) * train_ratio)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        # Process training images
        for img_file in tqdm(train_files, desc="Processing training images"):
            self._copy_image_and_label(source_images_dir, source_labels_dir, img_file, 'train')
        
        # Process validation images
        for img_file in tqdm(val_files, desc="Processing validation images"):
            self._copy_image_and_label(source_images_dir, source_labels_dir, img_file, 'val')
    
    def _copy_image_and_label(self, source_images_dir, source_labels_dir, img_file, subset):
        """
        Copy image and its corresponding label file to the YOLO dataset directory
        """
        # Copy image file
        src_img_path = os.path.join(source_images_dir, img_file)
        dst_img_path = os.path.join(self.yolo_dir, 'images', subset, img_file)
        shutil.copy(src_img_path, dst_img_path)
        
        # Copy label file if it exists
        label_file = os.path.splitext(img_file)[0] + '.txt'
        src_label_path = os.path.join(source_labels_dir, label_file)
        
        if os.path.exists(src_label_path):
            dst_label_path = os.path.join(self.yolo_dir, 'labels', subset, label_file)
            shutil.copy(src_label_path, dst_label_path)
        else:
            print(f"Warning: Label file not found for {img_file}")
    
    def _validate_dataset(self):
        """
        Validate the dataset for any issues before training
        """
        # Check if all images have corresponding label files
        for subset in ['train', 'val']:
            images_dir = os.path.join(self.yolo_dir, 'images', subset)
            labels_dir = os.path.join(self.yolo_dir, 'labels', subset)
            
            image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            label_files = [f for f in os.listdir(labels_dir) if f.lower().endswith('.txt')]
            
            for img_file in image_files:
                label_file = os.path.splitext(img_file)[0] + '.txt'
                label_path = os.path.join(labels_dir, label_file)
                if label_file not in label_files:
                    print(f"Warning: Label file not found for {img_file} in {subset} set. Creating empty label file.")
                    # Create an empty label file
                    with open(label_path, 'w') as f:
                        pass
        
        print("Dataset validation completed successfully!")
    
    def _create_yaml_config(self):
        yaml_content = {
            'path': self.yolo_dir,
            'train': os.path.join('images', 'train'),
            'val': os.path.join('images', 'val'),
            'names': {
                0: 'auto',
                1: 'non-auto'
            }
        }
        
        with open(os.path.join(self.yolo_dir, 'dataset.yaml'), 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
    
    def train_model(self, epochs=50, batch_size=16, img_size=640):
        print("Training YOLO model...")
        
        self.model = YOLO('yolov8n.pt')
        
        results = self.model.train(
            data=os.path.join(self.yolo_dir, 'dataset.yaml'),
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            save=True,
            project=os.path.join(self.data_dir, 'runs'),
            name='rickshaw_detection',
            device='',
            workers=8,
            seed=0,
            patience=0,
            verbose=True,
        )
        
        print("Model training completed!")
        return results
    
    def evaluate_model(self):
        if self.model is None:
            # Load the best model if not already loaded
            model_path = os.path.join(self.data_dir, 'runs', 'rickshaw_detection', 'weights', 'best.pt')
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
            else:
                raise ValueError("No trained model found. Please train the model first.")
        
        val_results = self.model.val(data=os.path.join(self.yolo_dir, 'dataset.yaml'))
        
        return val_results
    
    def load_model(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(self.data_dir, 'runs', 'rickshaw_detection', 'weights', 'best.pt')
        
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
            print(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")
    
    def detect_rickshaw(self, image_path, conf_threshold=0.25):
        if self.model is None:
            self.load_model()
        
        results = self.model.predict(image_path, conf=conf_threshold)
        print(f"[DEBUG] Raw model results: {results}")
        
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        detections = []
        
        for result in results:
            print(f"[DEBUG] Result object: {result}")
            boxes = result.boxes
            print(f"[DEBUG] Boxes: {boxes}")
            
            if len(boxes) == 0:
                print("No rickshaw detected in the image.")
                return img, []
            
            # Define colors for different classes
            colors = [(0, 255, 0), (255, 0, 0)]  # Green for auto, Red for non-auto
            
            for box in boxes:
                print(f"[DEBUG] Box: {box}")
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                cls_id = int(box.cls[0].item())
                conf = box.conf[0].item()
                cls_name = self.classes[cls_id]
                print(f"[DEBUG] Detection: class={cls_name}, conf={conf}, bbox=({x1}, {y1}, {x2}, {y2})")
                
                # Use different colors based on class
                color = colors[cls_id % len(colors)]
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                label = f"{cls_name}: {conf:.2f}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                detections.append({
                    'class': cls_name,
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2)
                })
        
        return img, detections
    
    def visualize_detection(self, image_path, conf_threshold=0.03,save_path=None):
        """
        Visualize rickshaw detection results
        
        Args:
            image_path: Path to the input image
            conf_threshold: Confidence threshold for detection
            save_path: Path to save the output image (optional)
            
        Returns:
            detections: List of detection results
        """
        img, detections = self.detect_rickshaw(image_path, conf_threshold)
        
        # Create a figure without displaying it (to avoid Qt issues)
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        fig = plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis('off')
        
        if not detections:
            plt.title("No rickshaw detected")
        else:
            plt.title(f"Detected {len(detections)} rickshaw(s)")
        
        # Save the figure if a save path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Detection visualization saved to {save_path}")
        
        plt.close(fig)  # Close the figure to avoid display
        
        return detections

def main():
    # Initialize the rickshaw detector
    detector = RickshawDetector()

    # Force full pipeline: dataset preparation, training, and evaluation
    print("Running full pipeline: dataset preparation, training, and evaluation.")
    
    # Prepare the dataset
    train_ratio = 0.9
    detector.prepare_dataset(train_ratio=train_ratio)
    print(f"Dataset split: Train ({train_ratio*100:.2f}%), Validation ({(1-train_ratio)*100:.2f}%)")
    
    # Train the model
    epochs = 50
    batch_size = 8
    detector.train_model(epochs=epochs, batch_size=batch_size)
    
    # Evaluate the model
    try:
        eval_results = detector.evaluate_model()
        print(f"Evaluation results: {eval_results}")
    except Exception as e:
        print(f"Evaluation error: {e}")

    # Example usage
    test_image = '/Users/dibbyoroy/Desktop/ML project/rickshaw_detection/Rickshaw-Detection/rickshaw_labeled_images/images/0d5f5251-849F7B9A-BA6A-40CB-A10C-A213F2DC38D0.jpg'
    if os.path.exists(test_image):
        # Save the detection visualization to a file
        output_path = os.path.join(os.path.dirname(test_image), 'detection_result.jpg')
        detections = detector.visualize_detection(test_image, conf_threshold=0.30,save_path=output_path)
        # Print detection results
        print(f"\nDetection Results:")
        for i, det in enumerate(detections):
            print(f"  {i+1}. {det['class']} (Confidence: {det['confidence']:.2f})")
            print(f"     Bounding Box: {det['bbox']}")
        print(f"\nTotal: {len(detections)} rickshaw(s) detected")
        print(f"\nVisualization saved to: {output_path}")

if __name__ == "__main__":
    main()
