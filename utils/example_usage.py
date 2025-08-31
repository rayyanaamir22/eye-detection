#!/usr/bin/env python3
"""
Example usage of the YOLO visualization utilities.

This script demonstrates how to use the visualization tools to:
1. Visualize single images with bounding boxes
2. Visualize dataset samples
3. Check dataset structure and validate labels
"""

import os
import sys
from visualization import YOLOVisualizer, visualize_single_image, visualize_dataset
from dataset_checker import print_dataset_report

def main():
    """Main function demonstrating the utilities."""
    
    # Define class names for eye detection
    class_names = ["left_eye", "right_eye"]
    
    print("YOLO Dataset Visualization Examples")
    print("=" * 50)
    
    # 1. Check dataset structure
    print("\n1. Checking dataset structure...")
    print_dataset_report("dataset")
    
    # 2. Example: Visualize a single image (uncomment when you have data)
    """
    print("\n2. Visualizing a single image...")
    image_path = "dataset/train/images/example.jpg"
    label_path = "dataset/train/labels/example.txt"
    
    if os.path.exists(image_path):
        visualize_single_image(
            image_path=image_path,
            label_path=label_path if os.path.exists(label_path) else None,
            class_names=class_names,
            show_labels=True
        )
    else:
        print(f"Example image not found: {image_path}")
    """
    
    # 3. Example: Visualize dataset samples (uncomment when you have data)
    """
    print("\n3. Visualizing dataset samples...")
    visualize_dataset(
        dataset_path="dataset",
        split="train",
        num_samples=3,
        class_names=class_names,
        show_labels=True
    )
    """
    
    # 4. Example: Using the YOLOVisualizer class directly
    print("\n4. Using YOLOVisualizer class...")
    visualizer = YOLOVisualizer(class_names)
    
    # Example usage (uncomment when you have data):
    """
    # Visualize single image
    visualizer.visualize_image(
        image_path="dataset/train/images/example.jpg",
        label_path="dataset/train/labels/example.txt",
        show_labels=True,
        save_path="output/visualization.png"
    )
    
    # Visualize dataset samples
    visualizer.visualize_dataset_sample(
        dataset_path="dataset",
        split="train",
        num_samples=5,
        show_labels=True
    )
    """
    
    print("\nExample completed!")
    print("\nTo use these utilities with your data:")
    print("1. Place your images in dataset/{train,val,test}/images/")
    print("2. Place your YOLO format labels in dataset/{train,val,test}/labels/")
    print("3. Uncomment the example code above")
    print("4. Run this script again")

def create_sample_data():
    """Create sample data for demonstration (optional)."""
    print("\nCreating sample data structure...")
    
    # Create sample directories if they don't exist
    sample_dirs = [
        "dataset/train/images",
        "dataset/train/labels", 
        "dataset/val/images",
        "dataset/val/labels",
        "dataset/test/images",
        "dataset/test/labels"
    ]
    
    for dir_path in sample_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created: {dir_path}")
    
    # Create a sample label file for demonstration
    sample_label_content = """0 0.5 0.3 0.2 0.15
1 0.7 0.3 0.2 0.15"""
    
    sample_label_path = "dataset/train/labels/sample.txt"
    with open(sample_label_path, 'w') as f:
        f.write(sample_label_content)
    
    print(f"Created sample label file: {sample_label_path}")
    print("Note: You'll need to add actual images to test the visualization")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--create-sample":
        create_sample_data()
    else:
        main()
