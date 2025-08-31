import cv2
import numpy as np
import os
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams

# Set matplotlib to use a non-interactive backend for better compatibility
rcParams['figure.figsize'] = (12, 8)

class YOLOVisualizer:
    """
    Utility class for visualizing images with YOLO format bounding boxes.
    
    YOLO format: <class> <x_center> <y_center> <width> <height>
    All values are normalized between 0 and 1.
    """
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Initialize the visualizer with optional class names.
        
        Args:
            class_names: List of class names corresponding to class indices
        """
        self.class_names = class_names or []
        # Default colors for different classes
        self.colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (128, 0, 0),    # Dark Red
            (0, 128, 0),    # Dark Green
            (0, 0, 128),    # Dark Blue
            (128, 128, 0),  # Olive
        ]
    
    def yolo_to_pixel_coords(self, yolo_box: List[float], img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """
        Convert YOLO format bounding box to pixel coordinates.
        
        Args:
            yolo_box: [class, x_center, y_center, width, height] in normalized coordinates
            img_width: Image width in pixels
            img_height: Image height in pixels
            
        Returns:
            Tuple of (x_min, y_min, x_max, y_max) in pixel coordinates
        """
        class_id, x_center, y_center, width, height = yolo_box
        
        # Convert normalized coordinates to pixel coordinates
        x_center_px = x_center * img_width
        y_center_px = y_center * img_height
        width_px = width * img_width
        height_px = height * img_height
        
        # Calculate corner coordinates
        x_min = int(x_center_px - width_px / 2)
        y_min = int(y_center_px - height_px / 2)
        x_max = int(x_center_px + width_px / 2)
        y_max = int(y_center_px + height_px / 2)
        
        return x_min, y_min, x_max, y_max
    
    def load_yolo_annotations(self, label_path: str) -> List[List[float]]:
        """
        Load YOLO format annotations from a text file.
        
        Args:
            label_path: Path to the label file
            
        Returns:
            List of bounding boxes in YOLO format
        """
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        values = [float(x) for x in line.split()]
                        if len(values) == 5:  # class, x_center, y_center, width, height
                            boxes.append(values)
        return boxes
    
    def visualize_image(self, image_path: str, label_path: Optional[str] = None, 
                       show_labels: bool = True, save_path: Optional[str] = None) -> None:
        """
        Visualize an image with optional bounding boxes.
        
        Args:
            image_path: Path to the image file
            label_path: Path to the corresponding label file (optional)
            show_labels: Whether to show class labels on bounding boxes
            save_path: Path to save the visualization (optional)
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return
        
        # Convert BGR to RGB for matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        
        # Create figure
        fig, ax = plt.subplots(1, 1)
        ax.imshow(image_rgb)
        
        # Load and draw bounding boxes if label file is provided
        if label_path:
            boxes = self.load_yolo_annotations(label_path)
            
            for box in boxes:
                class_id = int(box[0])
                x_min, y_min, x_max, y_max = self.yolo_to_pixel_coords(box, width, height)
                
                # Get color for this class
                color = self.colors[class_id % len(self.colors)]
                color_rgb = tuple(c/255 for c in color)
                
                # Create rectangle patch
                rect = patches.Rectangle(
                    (x_min, y_min), x_max - x_min, y_max - y_min,
                    linewidth=2, edgecolor=color_rgb, facecolor='none'
                )
                ax.add_patch(rect)
                
                # Add label if requested
                if show_labels:
                    label = f"Class {class_id}"
                    if class_id < len(self.class_names):
                        label = self.class_names[class_id]
                    
                    ax.text(x_min, y_min - 5, label, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=color_rgb, alpha=0.7),
                           fontsize=10, color='white', weight='bold')
        
        ax.set_title(f"Image: {os.path.basename(image_path)}")
        ax.axis('off')
        
        # Save if path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def visualize_dataset_sample(self, dataset_path: str, split: str = 'train', 
                                num_samples: int = 5, show_labels: bool = True) -> None:
        """
        Visualize a sample of images from the dataset.
        
        Args:
            dataset_path: Path to the dataset root directory
            split: Dataset split ('train', 'val', 'test')
            num_samples: Number of samples to visualize
            show_labels: Whether to show class labels
        """
        images_dir = os.path.join(dataset_path, 'images', split)
        labels_dir = os.path.join(dataset_path, 'labels', split)
        
        if not os.path.exists(images_dir):
            print(f"Error: Images directory not found at {images_dir}")
            return
        
        # Get list of image files
        image_files = [f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not image_files:
            print(f"No image files found in {images_dir}")
            return
        
        # Select random samples
        import random
        selected_files = random.sample(image_files, min(num_samples, len(image_files)))
        
        # Create subplot
        fig, axes = plt.subplots(1, len(selected_files), figsize=(5*len(selected_files), 5))
        if len(selected_files) == 1:
            axes = [axes]
        
        for i, image_file in enumerate(selected_files):
            image_path = os.path.join(images_dir, image_file)
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image_rgb.shape[:2]
            
            axes[i].imshow(image_rgb)
            
            # Load and draw bounding boxes
            if os.path.exists(label_path):
                boxes = self.load_yolo_annotations(label_path)
                
                for box in boxes:
                    class_id = int(box[0])
                    x_min, y_min, x_max, y_max = self.yolo_to_pixel_coords(box, width, height)
                    
                    color = self.colors[class_id % len(self.colors)]
                    color_rgb = tuple(c/255 for c in color)
                    
                    rect = patches.Rectangle(
                        (x_min, y_min), x_max - x_min, y_max - y_min,
                        linewidth=2, edgecolor=color_rgb, facecolor='none'
                    )
                    axes[i].add_patch(rect)
                    
                    if show_labels:
                        label = f"Class {class_id}"
                        if class_id < len(self.class_names):
                            label = self.class_names[class_id]
                        
                        axes[i].text(x_min, y_min - 5, label,
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color_rgb, alpha=0.7),
                                   fontsize=8, color='white', weight='bold')
            
            axes[i].set_title(f"{split}: {image_file}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()


def visualize_single_image(image_path: str, label_path: Optional[str] = None, 
                          class_names: Optional[List[str]] = None, 
                          show_labels: bool = True, save_path: Optional[str] = None) -> None:
    """
    Convenience function to visualize a single image.
    
    Args:
        image_path: Path to the image file
        label_path: Path to the corresponding label file (optional)
        class_names: List of class names
        show_labels: Whether to show class labels
        save_path: Path to save the visualization (optional)
    """
    visualizer = YOLOVisualizer(class_names)
    visualizer.visualize_image(image_path, label_path, show_labels, save_path)


def visualize_dataset(dataset_path: str, split: str = 'train', 
                     num_samples: int = 5, class_names: Optional[List[str]] = None,
                     show_labels: bool = True) -> None:
    """
    Convenience function to visualize dataset samples.
    
    Args:
        dataset_path: Path to the dataset root directory
        split: Dataset split ('train', 'val', 'test')
        num_samples: Number of samples to visualize
        class_names: List of class names
        show_labels: Whether to show class labels
    """
    visualizer = YOLOVisualizer(class_names)
    visualizer.visualize_dataset_sample(dataset_path, split, num_samples, show_labels)


# Example usage
if __name__ == "__main__":
    # Example class names for eye detection
    class_names = ["left_eye", "right_eye"]
    
    # Example usage
    visualizer = YOLOVisualizer(class_names)
    
    # Example: Visualize a single image
    # visualizer.visualize_image("path/to/image.jpg", "path/to/label.txt")
    
    # Example: Visualize dataset samples
    # visualizer.visualize_dataset_sample("dataset", "train", num_samples=3)
