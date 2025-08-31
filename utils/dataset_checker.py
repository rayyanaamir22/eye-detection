import os
import glob
from typing import Dict, List, Tuple
import yaml

def check_dataset_structure(dataset_path: str = "dataset") -> Dict:
    """
    Check the dataset structure and return statistics.
    
    Args:
        dataset_path: Path to the dataset root directory
        
    Returns:
        Dictionary containing dataset statistics and validation results
    """
    stats = {
        'splits': {},
        'total_images': 0,
        'total_labels': 0,
        'missing_labels': [],
        'orphaned_labels': [],
        'structure_valid': True,
        'errors': []
    }
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        images_dir = os.path.join(dataset_path, 'images', split)
        labels_dir = os.path.join(dataset_path, 'labels', split)
        
        split_stats = {
            'images_count': 0,
            'labels_count': 0,
            'missing_labels': [],
            'orphaned_labels': []
        }
        
        # Check if directories exist
        if not os.path.exists(images_dir):
            stats['errors'].append(f"Images directory not found: {images_dir}")
            stats['structure_valid'] = False
            continue
            
        if not os.path.exists(labels_dir):
            stats['errors'].append(f"Labels directory not found: {labels_dir}")
            stats['structure_valid'] = False
            continue
        
        # Count images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        images = []
        for ext in image_extensions:
            images.extend(glob.glob(os.path.join(images_dir, ext)))
            images.extend(glob.glob(os.path.join(images_dir, ext.upper())))
        
        split_stats['images_count'] = len(images)
        stats['total_images'] += len(images)
        
        # Count labels and check for missing/orphaned labels
        label_files = glob.glob(os.path.join(labels_dir, '*.txt'))
        split_stats['labels_count'] = len(label_files)
        stats['total_labels'] += len(label_files)
        
        # Check for missing labels
        for img_path in images:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(labels_dir, f"{img_name}.txt")
            
            if not os.path.exists(label_path):
                split_stats['missing_labels'].append(img_name)
                stats['missing_labels'].append(f"{split}/{img_name}")
        
        # Check for orphaned labels
        for label_path in label_files:
            label_name = os.path.splitext(os.path.basename(label_path))[0]
            img_path = os.path.join(images_dir, f"{label_name}.jpg")
            img_path_png = os.path.join(images_dir, f"{label_name}.png")
            
            if not os.path.exists(img_path) and not os.path.exists(img_path_png):
                split_stats['orphaned_labels'].append(label_name)
                stats['orphaned_labels'].append(f"{split}/{label_name}")
        
        stats['splits'][split] = split_stats
    
    return stats

def validate_yolo_labels(dataset_path: str = "dataset") -> Dict:
    """
    Validate YOLO format labels in the dataset.
    
    Args:
        dataset_path: Path to the dataset root directory
        
    Returns:
        Dictionary containing validation results
    """
    validation_results = {
        'valid_files': 0,
        'invalid_files': 0,
        'errors': [],
        'class_distribution': {}
    }
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        labels_dir = os.path.join(dataset_path, 'labels', split)
        
        if not os.path.exists(labels_dir):
            continue
        
        label_files = glob.glob(os.path.join(labels_dir, '*.txt'))
        
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                file_valid = True
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        values = [float(x) for x in line.split()]
                        
                        # Check if we have exactly 5 values
                        if len(values) != 5:
                            validation_results['errors'].append(
                                f"{split}/{os.path.basename(label_file)}: Line {line_num} - "
                                f"Expected 5 values, got {len(values)}"
                            )
                            file_valid = False
                            continue
                        
                        class_id, x_center, y_center, width, height = values
                        
                        # Check class ID
                        if not (0 <= class_id < 2):  # Assuming 2 classes for eye detection
                            validation_results['errors'].append(
                                f"{split}/{os.path.basename(label_file)}: Line {line_num} - "
                                f"Invalid class ID: {class_id}"
                            )
                            file_valid = False
                        
                        # Check if coordinates are normalized (between 0 and 1)
                        for coord_name, coord_value in [('x_center', x_center), ('y_center', y_center), 
                                                       ('width', width), ('height', height)]:
                            if not (0 <= coord_value <= 1):
                                validation_results['errors'].append(
                                    f"{split}/{os.path.basename(label_file)}: Line {line_num} - "
                                    f"{coord_name} not normalized: {coord_value}"
                                )
                                file_valid = False
                        
                        # Update class distribution
                        class_id_int = int(class_id)
                        if class_id_int not in validation_results['class_distribution']:
                            validation_results['class_distribution'][class_id_int] = 0
                        validation_results['class_distribution'][class_id_int] += 1
                        
                    except ValueError as e:
                        validation_results['errors'].append(
                            f"{split}/{os.path.basename(label_file)}: Line {line_num} - "
                            f"Invalid number format: {line}"
                        )
                        file_valid = False
                
                if file_valid:
                    validation_results['valid_files'] += 1
                else:
                    validation_results['invalid_files'] += 1
                    
            except Exception as e:
                validation_results['errors'].append(
                    f"{split}/{os.path.basename(label_file)}: Error reading file - {str(e)}"
                )
                validation_results['invalid_files'] += 1
    
    return validation_results

def print_dataset_report(dataset_path: str = "dataset"):
    """
    Print a comprehensive report about the dataset.
    
    Args:
        dataset_path: Path to the dataset root directory
    """
    print("=" * 60)
    print("DATASET STRUCTURE REPORT")
    print("=" * 60)
    
    # Check structure
    structure_stats = check_dataset_structure(dataset_path)
    
    print(f"\nDataset Path: {os.path.abspath(dataset_path)}")
    print(f"Structure Valid: {'✓' if structure_stats['structure_valid'] else '✗'}")
    
    if not structure_stats['structure_valid']:
        print("\nStructure Errors:")
        for error in structure_stats['errors']:
            print(f"  - {error}")
    
    print(f"\nOverall Statistics:")
    print(f"  Total Images: {structure_stats['total_images']}")
    print(f"  Total Labels: {structure_stats['total_labels']}")
    
    print(f"\nSplit Statistics:")
    for split, stats in structure_stats['splits'].items():
        print(f"  {split.upper()}:")
        print(f"    Images: {stats['images_count']}")
        print(f"    Labels: {stats['labels_count']}")
        print(f"    Missing Labels: {len(stats['missing_labels'])}")
        print(f"    Orphaned Labels: {len(stats['orphaned_labels'])}")
    
    if structure_stats['missing_labels']:
        print(f"\nMissing Labels ({len(structure_stats['missing_labels'])}):")
        for missing in structure_stats['missing_labels'][:10]:  # Show first 10
            print(f"  - {missing}")
        if len(structure_stats['missing_labels']) > 10:
            print(f"  ... and {len(structure_stats['missing_labels']) - 10} more")
    
    if structure_stats['orphaned_labels']:
        print(f"\nOrphaned Labels ({len(structure_stats['orphaned_labels'])}):")
        for orphaned in structure_stats['orphaned_labels'][:10]:  # Show first 10
            print(f"  - {orphaned}")
        if len(structure_stats['orphaned_labels']) > 10:
            print(f"  ... and {len(structure_stats['orphaned_labels']) - 10} more")
    
    # Validate labels
    print("\n" + "=" * 60)
    print("LABEL VALIDATION REPORT")
    print("=" * 60)
    
    validation_results = validate_yolo_labels(dataset_path)
    
    print(f"\nLabel Validation:")
    print(f"  Valid Files: {validation_results['valid_files']}")
    print(f"  Invalid Files: {validation_results['invalid_files']}")
    
    if validation_results['class_distribution']:
        print(f"\nClass Distribution:")
        class_names = ['left_eye', 'right_eye']
        for class_id, count in sorted(validation_results['class_distribution'].items()):
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
            print(f"  {class_name} (ID {class_id}): {count} instances")
    
    if validation_results['errors']:
        print(f"\nValidation Errors ({len(validation_results['errors'])}):")
        for error in validation_results['errors'][:10]:  # Show first 10
            print(f"  - {error}")
        if len(validation_results['errors']) > 10:
            print(f"  ... and {len(validation_results['errors']) - 10} more")
    
    print("\n" + "=" * 60)

def check_data_yaml(yaml_path: str = "dataset/data.yaml") -> bool:
    """
    Check if the data.yaml file is valid and matches the dataset structure.
    
    Args:
        yaml_path: Path to the data.yaml file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"\nData YAML Configuration:")
        print(f"  Path: {config.get('path', 'Not specified')}")
        print(f"  Train: {config.get('train', 'Not specified')}")
        print(f"  Val: {config.get('val', 'Not specified')}")
        print(f"  Test: {config.get('test', 'Not specified')}")
        print(f"  Classes: {config.get('nc', 'Not specified')}")
        print(f"  Class Names: {config.get('names', 'Not specified')}")
        
        return True
        
    except Exception as e:
        print(f"Error reading data.yaml: {e}")
        return False

if __name__ == "__main__":
    # Print comprehensive dataset report
    print_dataset_report()
    
    # Check data.yaml
    check_data_yaml()
