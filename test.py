**Title: Person Re-Identification (ReID) Dataset Preparation and Analysis**

---

## **Slide 1: Introduction**

### **What is Person Re-Identification (ReID)?**

- ReID is the task of matching images of the same person across different cameras.
- Commonly used in surveillance, autonomous vehicles, and smart cities.
- Requires well-structured datasets for training and evaluation.

### **Objective of this Presentation**

- Guide through dataset preparation from raw videos.
- Explain dataset splitting (train, query, gallery) across multiple cameras.
- Perform dataset analysis and validation.

---

## **Slide 2: Dataset Preparation Workflow**

1. **Extract Frames from Multi-Camera Videos (Aligned with 10 FPS MOT Labels)**
   - Convert each video into image frames at 10 FPS.
   - Maintain frame numbering for consistency.
   - Assign camera labels to extracted frames.
2. **Process MOT Ground Truth Labels for Multi-Camera Data**
   - Use bounding box annotations to crop person images per camera.
   - Assign unique person IDs while maintaining consistency across cameras.
3. **Generate Train, Query, and Gallery Splits**
   - Ensure proper identity separation across different camera views.
   - Balance data distribution.
4. **Format Dataset in Market-1501 Style**
   - Use structured naming conventions with camera information.
5. **Validate and Analyze Dataset**
   - Check identity distribution, camera balance, and cross-camera visibility.

---

## **Slide 3: Extracting Frames from Multi-Camera Videos (10 FPS)**

### **Steps:**

- Load each video using OpenCV.
- Extract frames at **10 FPS** (matching MOT annotations).
- Save frames with camera-specific identifiers.

### **Code Example:**

```python
import cv2
import os

def extract_frames(video_paths, output_folder, fps=10):
    os.makedirs(output_folder, exist_ok=True)
    
    for cam_id, video_path in enumerate(video_paths):
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / fps)  # Extract frames every (video_fps / 10) frames
        
        frame_count = 0
        saved_frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                frame_name = f"cam{cam_id}_{saved_frame_count:06d}.jpg"
                cv2.imwrite(os.path.join(output_folder, frame_name), frame)
                saved_frame_count += 1
            frame_count += 1
        cap.release()
```

---

## **Slide 4: Cropping Persons Using MOT Ground Truth (Multi-Camera, 10 FPS Alignment)**

### **MOT Format (Per Camera):**

```
frame, ID, x, y, width, height, conf, class, visibility
1, 1, 100, 200, 50, 100, 1, -1, 1
1, 2, 150, 250, 60, 120, 1, -1, 1
```

### **Steps:**

- Read MOT ground truth files for each camera.
- Crop bounding boxes from **10 FPS-aligned** frames per camera.
- Maintain consistent person IDs across different camera views.
- Save cropped images with camera-specific labels.

import pandas as pd
import cv2
import os

def crop_persons(mot_gt_file, frames_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    df = pd.read_csv(mot_gt_file, header=None)
    df.columns = ['frame', 'ID', 'x', 'y', 'w', 'h', 'conf', 'class', 'visibility']
    
    for _, row in df.iterrows():
        frame_path = os.path.join(frames_folder, f"{int(row['frame']):06d}.jpg")
        if not os.path.exists(frame_path):
            continue

        img = cv2.imread(frame_path)
        x, y, w, h = map(int, [row['x'], row['y'], row['w'], row['h']])
        person_crop = img[y:y+h, x:x+w]

        person_id = int(row['ID'])
        frame_id = int(row['frame'])
        person_folder = os.path.join(output_folder, f"{person_id:04d}")
        os.makedirs(person_folder, exist_ok=True)

        img_name = f"{frame_id:06d}.jpg"
        cv2.imwrite(os.path.join(person_folder, img_name), person_crop)

# Example usage
crop_persons("path/to/mot_gt.txt", "output/frames", "output/persons")


---
import random
import shutil

def create_splits(persons_folder, output_folder, train_ratio=0.5, query_ratio=0.2):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(f"{output_folder}/train", exist_ok=True)
    os.makedirs(f"{output_folder}/query", exist_ok=True)
    os.makedirs(f"{output_folder}/gallery", exist_ok=True)

    persons = os.listdir(persons_folder)
    random.shuffle(persons)

    num_train = int(len(persons) * train_ratio)
    train_persons = persons[:num_train]
    test_persons = persons[num_train:]

    for person in train_persons:
        shutil.move(os.path.join(persons_folder, person), f"{output_folder}/train/{person}")

    for person in test_persons:
        images = os.listdir(os.path.join(persons_folder, person))
        random.shuffle(images)

        num_query = int(len(images) * query_ratio)
        query_images = images[:num_query]
        gallery_images = images[num_query:]

        os.makedirs(f"{output_folder}/query/{person}", exist_ok=True)
        os.makedirs(f"{output_folder}/gallery/{person}", exist_ok=True)

        for img in query_images:
            shutil.move(os.path.join(persons_folder, person, img), f"{output_folder}/query/{person}/{img}")

        for img in gallery_images:
            shutil.move(os.path.join(persons_folder, person, img), f"{output_folder}/gallery/{person}/{img}")

# Example usage
create_splits("output/persons", "output/reid_splits")



## **Slide 5: Splitting Dataset (Train, Query, Gallery) for Multi-Camera**

### **Dataset Splitting Strategy:**

- **Train (50-60%)**: Used for training models.
- **Query (10-20%)**: Used for retrieval.
- **Gallery (30-40%)**: Used for evaluation.

### **Ensuring Identity Separation Across Cameras:**

- No overlapping IDs between train and test sets.
- Maintain cross-camera identity consistency.
- Ensure query and gallery images come from different cameras.

### **Code Example:**

```python
import random
import shutil

def create_splits(persons_folder, output_folder, train_ratio=0.5, query_ratio=0.2):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(f"{output_folder}/train", exist_ok=True)
    os.makedirs(f"{output_folder}/query", exist_ok=True)
    os.makedirs(f"{output_folder}/gallery", exist_ok=True)
    
    persons = os.listdir(persons_folder)
    random.shuffle(persons)
    
    num_train = int(len(persons) * train_ratio)
    train_persons = persons[:num_train]
    test_persons = persons[num_train:]
    
    for person in train_persons:
        shutil.move(os.path.join(persons_folder, person), f"{output_folder}/train/{person}")
    
    for person in test_persons:
        images = os.listdir(os.path.join(persons_folder, person))
        random.shuffle(images)
        
        num_query = int(len(images) * query_ratio)
        query_images = images[:num_query]
        gallery_images = images[num_query:]
        
        os.makedirs(f"{output_folder}/query/{person}", exist_ok=True)
        os.makedirs(f"{output_folder}/gallery/{person}", exist_ok=True)
        
        for img in query_images:
            shutil.move(os.path.join(persons_folder, person, img), f"{output_folder}/query/{person}/{img}")
        
        for img in gallery_images:
            shutil.move(os.path.join(persons_folder, person, img), f"{output_folder}/gallery/{person}/{img}")
```

---

import os

def rename_files(root_folder, cam_id=1):
    for subset in ["train", "query", "gallery"]:
        subset_path = os.path.join(root_folder, subset)
        for person in os.listdir(subset_path):
            person_path = os.path.join(subset_path, person)
            for img in os.listdir(person_path):
                frame_index = img.split('.')[0]
                new_name = f"{person}_{cam_id}_{frame_index}.jpg"
                os.rename(os.path.join(person_path, img), os.path.join(person_path, new_name))

# Example usage
rename_files("output/reid_splits")



## **Slide 6: Dataset Analysis - Identity & Camera Distribution**

- Check how many images each identity has.
- Analyze how many identities appear in multiple cameras.
- Visualize distribution with a histogram.

### **Code Example:**

import os
import matplotlib.pyplot as plt

def dataset_stats(dataset_path):
    subsets = ["train", "query", "gallery"]
    stats = {}

    for subset in subsets:
        subset_path = os.path.join(dataset_path, subset)
        person_counts = [len(os.listdir(os.path.join(subset_path, p))) for p in os.listdir(subset_path)]
        stats[subset] = person_counts

        plt.hist(person_counts, bins=20, alpha=0.5, label=subset)

    plt.xlabel("Number of images per identity")
    plt.ylabel("Count")
    plt.legend()
    plt.title("Distribution of images per identity")
    plt.show()

# Example usage
dataset_stats("output/reid_splits")


---

## **Slide 7-9: Additional Analysis & Conclusion**

- Identity Overlap Across Cameras.
- Camera Distribution per Identity.
- Next Steps for Model Training.



from torchreid import datasets

dataset = datasets.create('market1501', root='output/reid_splits', split_id=0)
print(dataset)
----------------------------------------

# stats
import os
import pandas as pd

# Path to the labels directory
labels_dir = "path/to/labels"

# Initialize sets and counters
unique_ids = set()
total_detections = 0

# Process each MOT label file in the directory
for label_file in os.listdir(labels_dir):
    if label_file.endswith(".txt"):  # Ensure processing only text files
        file_path = os.path.join(labels_dir, label_file)
        
        # Load the MOT format file (assuming space/comma-separated values)
        df = pd.read_csv(file_path, header=None, delim_whitespace=True)
        
        # MOT Format: frame, ID, x, y, w, h, conf, class, visibility
        person_ids = df[1].unique()  # Extract unique person IDs
        unique_ids.update(person_ids)  # Add to the global set
        total_detections += len(df)  # Count total bounding boxes

# Print statistics
print(f"Total Unique Person IDs: {len(unique_ids)}")
print(f"Total Detections (bounding boxes): {total_detections}")

-----------------------------------------\
import os
import pandas as pd
import numpy as np
import cv2
import glob

# Paths (Modify these based on your dataset)
labels_dir = "path/to/labels"
videos_dir = "path/to/videos"
output_screenshots_dir = "output/screenshots"

# Ensure output directory exists
os.makedirs(output_screenshots_dir, exist_ok=True)

# Initialize dictionaries for statistics
camera_stats = {}
person_camera_counts = {}

# Process each MOT label file
for label_file in os.listdir(labels_dir):
    if label_file.endswith(".txt"):  
        camera_id = os.path.splitext(label_file)[0]  # Assuming file name is camera ID
        file_path = os.path.join(labels_dir, label_file)

        # Read the label file
        df = pd.read_csv(file_path, header=None, delim_whitespace=True)

        # Extract person IDs and detection counts
        person_counts = df[1].value_counts().to_dict()  # {person_id: count}
        unique_person_ids = set(person_counts.keys())

        # Store stats per camera
        camera_stats[camera_id] = {
            "unique_person_count": len(unique_person_ids),
            "total_detections": len(df),
            "min_detections": min(person_counts.values()),
            "max_detections": max(person_counts.values()),
            "mean_detections": np.mean(list(person_counts.values())),
            "std_detections": np.std(list(person_counts.values())),
        }

        # Track how many times each person appears across different cameras
        for person_id in unique_person_ids:
            if person_id not in person_camera_counts:
                person_camera_counts[person_id] = set()
            person_camera_counts[person_id].add(camera_id)

# Compute overall statistics
num_cameras_per_person = [len(cameras) for cameras in person_camera_counts.values()]
average_appearances_across_cameras = np.mean(num_cameras_per_person)

# Convert results into a DataFrame for better visualization
camera_stats_df = pd.DataFrame.from_dict(camera_stats, orient="index")
print(camera_stats_df)

print(f"\nAverage number of times each ID appears across multiple cameras: {average_appearances_across_cameras}")

# Extract a sample frame for each camera
video_files = glob.glob(os.path.join(videos_dir, "*.mp4"))  # Adjust extension if needed
for video_file in video_files:
    camera_id = os.path.splitext(os.path.basename(video_file))[0]  # Match with label file
    cap = cv2.VideoCapture(video_file)
    
    if cap.isOpened():
        ret, frame = cap.read()  # Read the first frame
        if ret:
            output_path = os.path.join(output_screenshots_dir, f"{camera_id}.jpg")
            cv2.imwrite(output_path, frame)  # Save the frame as an image
    
    cap.release()

print("\nScreenshots saved in:", output_screenshots_dir)
-------------------------------------------------------------------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load previously computed camera statistics
camera_stats_df = pd.DataFrame.from_dict(camera_stats, orient="index")

# Set plot style
sns.set_style("whitegrid")

# Create output directory for plots
output_plots_dir = "output/plots"
os.makedirs(output_plots_dir, exist_ok=True)

### Plot 1: Unique Person IDs per Camera (Bar Chart)
plt.figure(figsize=(12, 6))
sns.barplot(x=camera_stats_df.index, y=camera_stats_df["unique_person_count"], palette="viridis")
plt.xticks(rotation=90)
plt.xlabel("Camera ID")
plt.ylabel("Unique Person IDs")
plt.title("Unique Person Count per Camera")
plt.savefig(os.path.join(output_plots_dir, "unique_persons_per_camera.png"))
plt.show()

### Plot 2: Histogram of Total Detections per Camera
plt.figure(figsize=(10, 5))
sns.histplot(camera_stats_df["total_detections"], bins=15, kde=True, color="blue")
plt.xlabel("Total Detections")
plt.ylabel("Frequency")
plt.title("Distribution of Total Detections per Camera")
plt.savefig(os.path.join(output_plots_dir, "detections_histogram.png"))
plt.show()

### Plot 3: Box Plot of Detections Per Person Across Cameras
plt.figure(figsize=(10, 5))
sns.boxplot(data=camera_stats_df[["min_detections", "max_detections", "mean_detections"]])
plt.xlabel("Detection Statistics")
plt.ylabel("Number of Detections")
plt.title("Variation in Detections Per Person Across Cameras")
plt.savefig(os.path.join(output_plots_dir, "detections_boxplot.png"))
plt.show()

### Plot 4: Scatter Plot - Unique IDs vs Total Detections per Camera
plt.figure(figsize=(10, 5))
sns.scatterplot(x=camera_stats_df["unique_person_count"], y=camera_stats_df["total_detections"], color="red", s=100)
plt.xlabel("Unique Person IDs")
plt.ylabel("Total Detections")
plt.title("Unique IDs vs Total Detections per Camera")
plt.savefig(os.path.join(output_plots_dir, "scatter_unique_vs_detections.png"))
plt.show()

### Plot 5: Heatmap of Correlations Between Statistics
plt.figure(figsize=(8, 6))
sns.heatmap(camera_stats_df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap of Camera Statistics")
plt.savefig(os.path.join(output_plots_dir, "correlation_heatmap.png"))
plt.show()

print(f"\nPlots saved in: {output_plots_dir}")

