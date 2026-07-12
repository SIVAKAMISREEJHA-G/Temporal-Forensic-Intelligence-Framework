import os
import sys
import glob
import cv2
import pandas as pd
import numpy as np

# Set standard paths
BASE_DIR = r"C:\Users\sivak\Downloads\crime data"
OUTPUT_DIR = os.path.join(BASE_DIR, "TFIF", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "dataset_inspection_summary.txt")

FOLDER_MAPPING = {
    "abuse6": "Abuse",
    "car accident40": "Car Accident",
    "explosion40": "Explosion",
    "fighting40": "Fighting",
    "normal50": "Normal",
    "riot40": "Riot",
    "shooting40": "Shooting"
}

def rename_folders():
    print("Checking folders for renaming...")
    renamed = False
    for old_name, new_name in FOLDER_MAPPING.items():
        old_path = os.path.join(BASE_DIR, old_name)
        new_path = os.path.join(BASE_DIR, new_name)
        if os.path.exists(old_path) and not os.path.exists(new_path):
            print(f"Renaming '{old_name}' -> '{new_name}'")
            os.rename(old_path, new_path)
            renamed = True
        elif os.path.exists(old_path) and os.path.exists(new_path):
            # Move files from old to new if both exist
            print(f"Both old and new paths exist. Merging '{old_name}' into '{new_name}'")
            for f in os.listdir(old_path):
                src = os.path.join(old_path, f)
                dst = os.path.join(new_path, f)
                if not os.path.exists(dst):
                    os.rename(src, dst)
            try:
                os.rmdir(old_path)
            except Exception as e:
                print(f"Could not remove old folder {old_name}: {e}")
            renamed = True
    if renamed:
        print("Folder renaming complete.")
    else:
        print("No folders needed renaming or renaming already completed.")

def inspect_dataset():
    rename_folders()
    
    # We will look for all folders in FOLDER_MAPPING values
    classes = list(FOLDER_MAPPING.values())
    video_extensions = ["*.mp4", "*.avi", "*.mkv", "*.mov"]
    
    records = []
    corrupted_files = []
    
    print("\nScanning videos and computing metadata...")
    for cls in classes:
        cls_dir = os.path.join(BASE_DIR, cls)
        if not os.path.exists(cls_dir):
            print(f"Warning: Folder '{cls}' not found at {cls_dir}")
            continue
            
        video_paths = []
        for ext in video_extensions:
            video_paths.extend(glob.glob(os.path.join(cls_dir, ext)))
            
        print(f"Found {len(video_paths)} videos in class '{cls}'")
        for v_path in video_paths:
            v_name = os.path.basename(v_path)
            cap = cv2.VideoCapture(v_path)
            
            if not cap.isOpened():
                print(f"  [CORRUPTED] Cannot open {v_name}")
                corrupted_files.append((cls, v_name, v_path, "Could not open file"))
                cap.release()
                continue
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if frame_count <= 0 or fps <= 0 or width <= 0 or height <= 0:
                print(f"  [CORRUPTED] Invalid metadata for {v_name} (frames={frame_count}, fps={fps}, size={width}x{height})")
                corrupted_files.append((cls, v_name, v_path, f"Invalid metadata: frame_count={frame_count}, fps={fps}"))
                cap.release()
                continue
                
            # Attempt to read first frame to test readability
            ret, _ = cap.read()
            if not ret:
                print(f"  [CORRUPTED] Cannot read first frame of {v_name}")
                corrupted_files.append((cls, v_name, v_path, "Cannot read first frame"))
                cap.release()
                continue
                
            duration = frame_count / fps
            records.append({
                "class": cls,
                "video_name": v_name,
                "duration": duration,
                "fps": fps,
                "frames": frame_count,
                "resolution": f"{width}x{height}",
                "width": width,
                "height": height
            })
            cap.release()
            
    df = pd.DataFrame(records)
    
    # Compile report text
    report = []
    report.append("="*60)
    report.append("TEMPORAL FORENSIC INTELLIGENCE FRAMEWORK (TFIF)")
    report.append("DATASET INSPECTION REPORT")
    report.append("="*60)
    report.append(f"Base Directory: {BASE_DIR}")
    report.append(f"Total Readable Videos Found: {len(df)}")
    report.append(f"Total Corrupted Videos Found: {len(corrupted_files)}\n")
    
    if corrupted_files:
        report.append("CORRUPTED/UNREADABLE FILES DETECTED:")
        for cls, name, path, reason in corrupted_files:
            report.append(f"  - [{cls}] {name} (Reason: {reason})")
        report.append("\n")
    else:
        report.append("No corrupted or unreadable video files detected.\n")
        
    report.append("PER-CLASS VIDEO STATS:")
    report.append("-" * 60)
    
    if len(df) > 0:
        class_summary = df.groupby("class").agg(
            Count=("video_name", "count"),
            MinDuration=("duration", "min"),
            MaxDuration=("duration", "max"),
            AvgDuration=("duration", "mean"),
            MinFPS=("fps", "min"),
            MaxFPS=("fps", "max"),
            AvgFPS=("fps", "mean"),
            Resolutions=("resolution", lambda x: ", ".join(x.unique()))
        ).reset_index()
        
        for idx, row in class_summary.iterrows():
            report.append(f"Class: {row['class']}")
            report.append(f"  - Video Count: {row['Count']}")
            report.append(f"  - Duration (seconds): Min = {row['MinDuration']:.2f}s, Max = {row['MaxDuration']:.2f}s, Avg = {row['AvgDuration']:.2f}s")
            report.append(f"  - Frame Rate (FPS): Min = {row['MinFPS']:.2f}, Max = {row['MaxFPS']:.2f}, Avg = {row['AvgFPS']:.2f}")
            report.append(f"  - Resolutions present: {row['Resolutions']}")
            report.append("-" * 30)
            
        # Overall Summary
        report.append("\nOVERALL DATASET STATISTICS:")
        report.append(f"  - Total Videos: {len(df)}")
        report.append(f"  - Duration: Min = {df['duration'].min():.2f}s, Max = {df['duration'].max():.2f}s, Avg = {df['duration'].mean():.2f}s")
        report.append(f"  - Class counts: {dict(df['class'].value_counts())}")
    else:
        report.append("No video data extracted successfully.")
        
    report_text = "\n".join(report)
    print(report_text)
    
    with open(SUMMARY_FILE, "w") as f:
        f.write(report_text)
    print(f"\nSummary successfully written to: {SUMMARY_FILE}")

if __name__ == "__main__":
    inspect_dataset()
