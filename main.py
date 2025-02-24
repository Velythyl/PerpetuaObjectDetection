import os
import shutil
import numpy as np
from ultralytics import YOLO
from datetime import datetime

# Load YOLO model once
model = YOLO("yolo11n.pt").cuda()

# Define the classes we're interested in
classes_of_interest = ['truck', 'car']

# Get the class IDs from the model
class_ids = {model.names[i]: i for i in model.names if model.names[i] in classes_of_interest}

if not class_ids:
    raise ValueError("None of the specified classes were found in the model's class names.")


def detect_objects(image_path, savedir):
    """Run YOLO inference on an image and return detected classes"""
    shutil.rmtree("/tmp/perpetua", ignore_errors=True)

    results = model(image_path, save=True, project="/tmp", name="perpetua")
    detected_classes = set()

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            class_name = model.names[class_id]
            if class_name in classes_of_interest:
                detected_classes.add(class_name)

    imagename = image_path.split("/")[-1]
    shutil.move("/tmp/perpetua/" + imagename, savedir + f"/{imagename}")

    return detected_classes


def create_presence_vector(detected_classes):
    """Create a binary presence vector based on detected classes"""
    return np.array([1 if cls in detected_classes else 0 for cls in classes_of_interest])


from datetime import datetime
from datetime import datetime


def parse_timestamp_from_filename(filename):
    """Parse timestamp from image filename using multiple common formats."""
    base_name = os.path.splitext(filename)[0]

    # List of common timestamp formats to try
    timestamp_formats = [
        "%Y-%m-%d_%H_%M_%S",  # YYYY-MM-DD_HH_MM_SS
        "%Y%m%d_%H%M%S",  # YYYYMMDD_HHMMSS
        "%Y-%m-%d %H:%M:%S",  # YYYY-MM-DD HH:MM:SS
        "%Y%m%d %H%M%S",  # YYYYMMDD HHMMSS
        "%Y-%m-%d_%H-%M-%S",  # YYYY-MM-DD_HH-MM-SS
        "%Y%m%d-%H%M%S",  # YYYYMMDD-HHMMSS
        "%Y-%m-%dT%H:%M:%S",  # YYYY-MM-DDTHH:MM:SS (ISO 8601)
        "%Y%m%dT%H%M%S",  # YYYYMMDDTHHMMSS
        "%Y%m%d%H%M%S",  # YYYYMMDDHHMMSS
        "%Y-%m-%d-%H%M%S",  # YYYY-MM-DD-HHMMSS (new format)
    ]

    for fmt in timestamp_formats:
        try:
            # Try to parse the base name directly
            dt = datetime.strptime(base_name, fmt)
            return dt.timestamp()
        except ValueError:
            continue

    # If none of the formats worked, try splitting the base name
    parts = base_name.split('_')
    if len(parts) >= 4:
        # Try formats that assume the date and time are separated by underscores
        date_time_formats = [
            "%Y-%m-%d %H:%M:%S",  # YYYY-MM-DD HH:MM:SS
            "%Y%m%d %H%M%S",  # YYYYMMDD HHMMSS
        ]
        date_str = parts[0]
        time_str = f"{parts[1]}:{parts[2]}:{parts[3]}"
        datetime_str = f"{date_str} {time_str}"

        for fmt in date_time_formats:
            try:
                dt = datetime.strptime(datetime_str, fmt)
                return dt.timestamp()
            except ValueError:
                continue

    # If all attempts fail, raise an error
    raise ValueError(f"Filename {filename} does not match any known timestamp format.")



def process_directory(image_dir, savedir):
    """Process all images in a directory and save results"""
    presence_vectors = []
    timestamps = []
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        try:
            # Parse timestamp from filename
            timestamp = parse_timestamp_from_filename(image_file)
            timestamps.append(timestamp)

            # Detect objects and create presence vector
            detected_classes = detect_objects(image_path, savedir)
            presence_vector = create_presence_vector(detected_classes)
            presence_vectors.append(presence_vector)
            print(f"Processed {image_file}: {presence_vector}")
        except ValueError as e:
            print(f"Skipping {image_file}: {str(e)}")
            continue

    # Convert lists to numpy arrays
    timestamps_array = np.array(timestamps, dtype=np.float64)
    presence_vectors_array = np.array(presence_vectors)

    # Combine timestamps and presence vectors into a structured array
    structured_array = np.column_stack((timestamps_array, presence_vectors_array))

    # Save the results
    output_path = os.path.join(image_dir, 'detection_results.npy')
    np.save(output_path, structured_array)
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    image_directory = "./converted"
    shutil.rmtree(os.path.join(image_directory, "detections"), ignore_errors=True)
    savedir = os.path.join(image_directory, "detections")
    os.makedirs(savedir)
    process_directory(image_directory, savedir)