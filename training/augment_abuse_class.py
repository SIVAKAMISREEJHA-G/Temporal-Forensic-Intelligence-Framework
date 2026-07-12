import os
import cv2
import numpy as np
import random

BASE_DIR = r"C:\Users\sivak\Downloads\crime data"
ABUSE_DIR = os.path.join(BASE_DIR, "Abuse")
TARGET_COUNT = 40

def add_gaussian_noise(frame, std=10):
    noise = np.random.normal(0, std, frame.shape).astype(np.float32)
    noisy_frame = frame.astype(np.float32) + noise
    return np.clip(noisy_frame, 0, 255).astype(np.uint8)

def adjust_brightness_contrast(frame, alpha=1.0, beta=0):
    # alpha: contrast, beta: brightness
    adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    return adjusted

def random_crop_resize(frame, crop_factor=0.9):
    h, w, c = frame.shape
    new_h, new_w = int(h * crop_factor), int(w * crop_factor)
    start_y = random.randint(0, h - new_h)
    start_x = random.randint(0, w - new_w)
    cropped = frame[start_y:start_y+new_h, start_x:start_x+new_w]
    return cv2.resize(cropped, (w, h))

def augment_video(source_path, target_path, aug_id):
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        print(f"Error: Cannot open source video {source_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Use MP4V codec which is standard and robust on Windows
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(target_path, fourcc, fps, (width, height))

    # Define augmentation parameters for this video
    flip_h = (aug_id % 2 == 0)
    adjust_bc = (aug_id % 3 == 0)
    crop = (aug_id % 4 == 0)
    noise = (aug_id % 5 == 0)
    speed_change = 0  # 0: normal, 1: slow (0.9x), 2: fast (1.1x)
    if aug_id % 6 == 0:
        speed_change = 1
    elif aug_id % 7 == 0:
        speed_change = 2

    alpha = random.uniform(0.85, 1.15)
    beta = random.randint(-15, 15)
    crop_factor = random.uniform(0.85, 0.95)

    frame_idx = 0
    written_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        
        # Apply speed change:
        # Slow down (0.9x): repeat frames occasionally
        # Speed up (1.1x): skip frames occasionally
        if speed_change == 1: # slow down (repeat every 9th frame)
            if frame_idx % 9 == 0:
                # We will process and write this frame twice
                frames_to_write = [frame, frame]
            else:
                frames_to_write = [frame]
        elif speed_change == 2: # speed up (skip every 10th frame)
            if frame_idx % 10 == 0:
                continue
            frames_to_write = [frame]
        else:
            frames_to_write = [frame]

        for frm in frames_to_write:
            processed = frm.copy()
            
            if flip_h:
                processed = cv2.flip(processed, 1)
            if adjust_bc:
                processed = adjust_brightness_contrast(processed, alpha, beta)
            if crop:
                processed = random_crop_resize(processed, crop_factor)
            if noise:
                processed = add_gaussian_noise(processed, std=random.uniform(5, 12))
                
            out.write(processed)
            written_frames += 1

    cap.release()
    out.release()
    print(f"Created augmented video {os.path.basename(target_path)} from {os.path.basename(source_path)} (frames written: {written_frames})")
    return True

def run_augmentation():
    # Find all original abuse videos
    source_videos = [
        os.path.join(ABUSE_DIR, f) for f in os.listdir(ABUSE_DIR)
        if f.endswith(".mp4") and not f.startswith("abuse_aug")
    ]
    
    num_source = len(source_videos)
    if num_source == 0:
        print("Error: No source Abuse videos found!")
        return

    print(f"Found {num_source} source Abuse videos.")
    num_to_generate = TARGET_COUNT - num_source
    print(f"Generating {num_to_generate} augmented videos to reach {TARGET_COUNT} total...")

    success_count = 0
    for i in range(1, num_to_generate + 1):
        # Pick source video round-robin
        src_path = source_videos[(i - 1) % num_source]
        target_name = f"abuse_aug_{i:02d}.mp4"
        target_path = os.path.join(ABUSE_DIR, target_name)
        
        if augment_video(src_path, target_path, i):
            success_count += 1
            
    print(f"Successfully generated {success_count} augmented videos in {ABUSE_DIR}")

if __name__ == "__main__":
    run_augmentation()
