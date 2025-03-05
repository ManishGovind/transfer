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
