import os
import argparse
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from rickshaw_detection import RickshawDetector

def main():
    parser = argparse.ArgumentParser(description='Rickshaw Detection Inference')
    parser.add_argument('--image', type=str, required=True, help='Path to the image for inference')
    parser.add_argument('--model', type=str, default=None, help='Path to the model weights (optional)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for detection')
    parser.add_argument('--save', type=str, default=None, help='Path to save the output image (optional)')
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        return
    
    detector = RickshawDetector()
    
    try:
        detector.load_model(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print(f"Running inference on {args.image}...")
    
    # Determine save path if not provided
    save_path = args.save
    if not save_path:
        base_name = os.path.basename(args.image)
        save_path = os.path.join(os.path.dirname(args.image), f"detection_{base_name}")
    
    # Use the detect_rickshaw method to get both image and detections
    img, detections = detector.detect_rickshaw(args.image, conf_threshold=args.conf)
    
    # Save the visualization manually with clear class labels
    if detections:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(1, figsize=(12, 10))
        ax.imshow(img)
        
        # Count auto and non-auto rickshaws
        auto_count = sum(1 for det in detections if det['class'] == 'auto')
        non_auto_count = sum(1 for det in detections if det['class'] == 'non-auto')
        
        # Add bounding boxes with clear labels
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            width = x2 - x1
            height = y2 - y1
            
            # Use different colors and styles for auto vs non-auto
            if det['class'] == 'auto':
                rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='green', facecolor='none')
                label = f"AUTO-RICKSHAW: {det['confidence']:.2f}"
                color = 'green'
            else:
                rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='red', facecolor='none')
                label = f"NON-AUTO RICKSHAW: {det['confidence']:.2f}"
                color = 'red'
                
            ax.add_patch(rect)
            ax.text(x1, y1-10, label, color=color, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
        
        ax.set_title(f"Detected {len(detections)} rickshaw(s): {auto_count} auto, {non_auto_count} non-auto")
        ax.axis('off')
        
        # Save the figure
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Detection visualization saved to {save_path}")
    else:
        print("No detections to visualize.")
    
    if not detections:
        print("No rickshaws detected in the image.")
    else:
        print(f"\nDetection Results:")
        auto_count = 0
        non_auto_count = 0
        
        for i, det in enumerate(detections):
            class_name = det['class']
            if class_name == 'auto':
                auto_count += 1
                class_display = "AUTO-RICKSHAW"
            else:
                non_auto_count += 1
                class_display = "NON-AUTO RICKSHAW"
                
            print(f"  {i+1}. {class_display} (Confidence: {det['confidence']:.2f})")
            print(f"     Bounding Box: {det['bbox']}")
        
        print(f"\nSummary:")
        print(f"  - AUTO-RICKSHAWS detected: {auto_count}")
        print(f"  - NON-AUTO RICKSHAWS detected: {non_auto_count}")
        print(f"  - TOTAL rickshaws detected: {len(detections)}")
        print(f"\nVisualization saved to: {save_path}")

if __name__ == "__main__":
    main()
