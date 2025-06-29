# Rickshaw Detection System

This project uses YOLOv8 to detect and classify auto-rickshaws and non-auto rickshaws in images.

## Features

- Detects auto-rickshaws (with engines) and non-auto rickshaws (manually paddled)
- Handles multiple objects in a single image
- Provides confidence scores for detections
- Visualizes detection results with color-coded bounding boxes (green for auto, red for non-auto)

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Dataset structure:
   - The system uses labeled images from the `rickshaw_labeled_images` folder
   - Images are stored in the `rickshaw_labeled_images/images` directory
   - Labels are stored in the `rickshaw_labeled_images/labels` directory in YOLO format

## Usage

Run the main script to prepare the dataset, train the model, and evaluate it:

```bash
python rickshaw_detection.py
```

### Custom Inference

To use the trained model for inference on new images, you can use the command-line interface or the Python API:

#### Command-line interface

```bash
python inference.py --image path/to/your/image.jpg --conf 0.25 --save output.jpg
```

Options:
- `--image`: Path to the input image (required)
- `--model`: Path to custom model weights (optional)
- `--conf`: Confidence threshold (default: 0.25)
- `--save`: Path to save the output image (optional)

#### Python API

```python
from rickshaw_detection import RickshawDetector

# Initialize detector and load trained model
detector = RickshawDetector()
detector.load_model()  # Will load the best model from the training run

# Detect rickshaws in an image
img, detections = detector.detect_rickshaw('path/to/your/image.jpg')

# Print detection results
for i, det in enumerate(detections):
    print(f"Detection {i+1}: {det['class']} with confidence {det['confidence']:.2f}")
```

## Model Training

The system automatically:
1. Processes the pre-labeled dataset in YOLO format
2. Splits data into training and validation sets
3. Trains a YOLOv8 model (default: 50 epochs)
4. Saves the best model weights
5. Evaluates model performance

## Label Format

The system uses YOLO format annotations where each line in the label file represents:
```
<class_id> <x_center> <y_center> <width> <height>
```

Where:
- `class_id`: 0 for auto-rickshaw, 1 for non-auto rickshaw
- `x_center`, `y_center`: Normalized center coordinates (0-1)
- `width`, `height`: Normalized width and height (0-1)

## Customization

You can adjust the following parameters in the code:
- Training epochs
- Batch size
- Image size
- Confidence threshold for detection
- Train/validation split ratio
