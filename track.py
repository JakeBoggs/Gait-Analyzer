#!/usr/bin/env python3
"""
track.py
----------------------------------------
Play video with space‑pause/resume; when paused (or stepping),
run YOLO11‑pose, compute 5 joint points + three angles,
overlay on the main frame, and allow drag & drop to adjust points.
Forward/back arrows instantly load the new frame and overlay.
Press 's' to save the cropped image with points and angles to the output directory.
"""
import argparse
import cv2
import numpy as np
import os
import datetime
import csv
from ultralytics import YOLO

# ----------------------------------------------------------------------
# COCO 17‑point mapping
# ----------------------------------------------------------------------
KP = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
}
MIN_SCORE = 0.2
PADDING = 0.2
DELAY_MS = 30

# ----------------------------------------------------------------------
# Globals
# ----------------------------------------------------------------------
points = []        # five draggable points in original coords
angles = [None]*3
bbox = None        # (x1, y1, x2, y2) padded
paused = False
detection_needed = False
last_frame = None
idx = 0
total = 0
dragging = False
drag_idx = -1
video_basename = ""
output_dir = ""
output_csv_path = ""

def angle_between(v1, v2):
    n1,n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1<1e-6 or n2<1e-6:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(n1*n2),-1,1))))

def recompute_angles():
    global angles, points
    S,H,K,A,T = [np.array(p) for p in points]
    angles[0] = 180 - angle_between(S-H, K-H)
    angles[1] = 180 - angle_between(H-K, A-K)
    angles[2] = 90 - angle_between(K-A, T-A)

def draw_main_window():
    """Overlay bbox, points, and frame number on last_frame."""
    global last_frame, bbox, points, angles, idx, video_basename
    vis = last_frame.copy()

    # Define colors for points (BGR): S, H, K, A, T
    point_colors = [(0, 255, 0),  # Green (Shoulder Center)
                    (255, 0, 0),  # Blue (Hip Center)
                    (0, 0, 255),  # Red (Knee)
                    (0, 255, 255), # Yellow (Ankle)
                    (255, 255, 0)] # Cyan (Toe)

    # Draw bbox if exists
    if bbox:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Draw points if exists
    if points:
        for i, (x, y) in enumerate(points):
            color = point_colors[i % len(point_colors)] # Use defined colors
            cv2.circle(vis, (int(x), int(y)), 6, (0, 0, 0), -1)  # black border
            cv2.circle(vis, (int(x), int(y)), 4, color, -1)  # colored dot

    # Draw frame number and angles in top-left corner
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    border_thickness = 6
    y_offset = 35
    x_position = 10

    # Draw Frame number
    frame_text = f"Frame: {idx}"
    frame_position = (x_position, y_offset)
    cv2.putText(vis, frame_text, frame_position, font, font_scale,
                (0, 0, 0), border_thickness, cv2.LINE_AA)
    cv2.putText(vis, frame_text, frame_position, font, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)

    # Draw Angles below frame number
    if bbox and angles and all(a is not None for a in angles):
        labels = ["S-H-K", "H-K-A", "K-A-T"]
        angle_y_start = y_offset + 35 # Start below frame number
        for i, angle in enumerate(angles):
            text = f"{labels[i]}: {angle:.1f}"
            position = (x_position, angle_y_start + i * 35)
            # Black border
            cv2.putText(vis, text, position, font, font_scale,
                        (0, 0, 0), border_thickness, cv2.LINE_AA)
            # White text
            cv2.putText(vis, text, position, font, font_scale,
                        (255, 255, 255), thickness, cv2.LINE_AA)

    cv2.imshow(video_basename, vis)

def mouse_cb(event, x, y, flags, param):
    """Drag points on the window."""
    global points, dragging, drag_idx
    if event == cv2.EVENT_LBUTTONDOWN:
        for i,(px,py) in enumerate(points):
            if (x-px)**2 + (y-py)**2 < 10**2:
                dragging, drag_idx = True, i
                return
    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        points[drag_idx] = (x,y)
        recompute_angles()
        draw_main_window()
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False

def run_detection(frame, model):
    """Return padded bbox+points or None."""
    r0 = model(frame, verbose=False, device="cpu")[0]
    k0 = r0.keypoints.data
    b0 = r0.boxes.xyxy.data
    if k0 is None or len(k0)==0:
        return None
    best0 = int(np.argmax(k0[:,:,2].sum(axis=1)))
    x1,y1,x2,y2 = map(int, b0[best0])
    h,w = frame.shape[:2]
    pad_w = int((x2-x1)*PADDING)
    pad_h = int((y2-y1)*PADDING)
    x1p,y1p = max(0,x1-pad_w), max(0,y1-pad_h)
    x2p,y2p = min(w,x2+pad_w), min(h,y2+pad_h)
    crop = frame[y1p:y2p, x1p:x2p]
    r1 = model(crop, verbose=False)[0]
    k1 = r1.keypoints.data
    if k1 is None or len(k1)==0:
        return None
    best1 = int(np.argmax(k1[:,:,2].sum(axis=1)))
    kps = k1[best1]
    Ls,Rs = kps[KP['left_shoulder']][:2], kps[KP['right_shoulder']][:2]
    S = ((Ls+Rs)/2).tolist()
    Lh,Rh = kps[KP['left_hip']][:2], kps[KP['right_hip']][:2]
    H = ((Lh+Rh)/2).tolist()
    side = 'left' if kps[KP['left_knee'],2] > kps[KP['right_knee'],2] else 'right'
    Kpt = kps[KP[f"{side}_knee"]][:2].tolist()
    Apt = kps[KP[f"{side}_ankle"]][:2].tolist()
    
    # Convert Kpt and Apt to numpy arrays for vector math
    Kpt_np = np.array(Kpt)
    Apt_np = np.array(Apt)

    # Calculate leg vector (Knee -> Ankle)
    leg_vector_ka = Apt_np - Kpt_np

    # Calculate perpendicular vector based on the side
    # Left leg: Rotate counter-clockwise (swap, negate first) -> (-dy, dx)
    # Right leg: Rotate clockwise (swap, negate second) -> (dy, -dx)
    if side == 'left':
        perp_vector = np.array([-leg_vector_ka[1], leg_vector_ka[0]])
    else: # side == 'right'
        perp_vector = np.array([leg_vector_ka[1], -leg_vector_ka[0]])

    # Calculate toe offset vector. The length factor 0.5 makes ||AT|| = 0.5 * ||KA||.
    toe_offset_factor = 0.5
    toe_offset_vector = toe_offset_factor * perp_vector

    # Calculate final Toe point T = Ankle + offset vector
    T_np = Apt_np + toe_offset_vector
    T = T_np.tolist() # Convert back to list

    pts = [(S[0]+x1p, S[1]+y1p),
           (H[0]+x1p, H[1]+y1p),
           (Kpt[0]+x1p, Kpt[1]+y1p),
           (Apt[0]+x1p, Apt[1]+y1p),
           (T[0]+x1p, min(T[1]+y1p, frame.shape[0] - 10))]
    return (x1p,y1p,x2p,y2p), pts

def save_cropped_image():
    global last_frame, bbox, points, angles, idx, video_basename, output_dir, output_csv_path
    if bbox is None:
        print("No detection to save.")
        return
    x1, y1, x2, y2 = bbox
    cropped = last_frame[y1:y2, x1:x2].copy()
    shifted_points = [(x - x1, y - y1) for (x, y) in points]

    # Define colors for points (BGR): S, H, K, A, T (same as draw_main_window)
    point_colors = [(0, 255, 0),  # Green (Shoulder Center)
                    (255, 0, 0),  # Blue (Hip Center)
                    (0, 0, 255),  # Red (Knee)
                    (0, 255, 255), # Yellow (Ankle)
                    (255, 255, 0)] # Cyan (Toe)

    # Draw points with defined colors
    if shifted_points:
        for i, (x, y) in enumerate(shifted_points):
            color = point_colors[i % len(point_colors)] # Use defined colors
            cv2.circle(cropped, (int(x), int(y)), 6, (0, 0, 0), -1) # black border
            cv2.circle(cropped, (int(x), int(y)), 4, color, -1) # colored dot

    labels = ["S-H-K", "H-K-A", "K-A-T"]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2
    border_thickness = 4
    y_offset = 30
    
    for i, angle in enumerate(angles):
        if angle is not None:
            text = f"{labels[i]}: {angle:.1f}"
            position = (5, y_offset + i * 35)

            # Black border (thicker)
            cv2.putText(cropped, text, position, font, font_scale, 
                        (0, 0, 0), border_thickness, cv2.LINE_AA)
            # White text on top
            cv2.putText(cropped, text, position, font, font_scale, 
                        (255, 255, 255), thickness, cv2.LINE_AA)

    # Save image to timestamped directory
    image_filename = os.path.join(output_dir, f"{video_basename}_{idx:04d}.png")
    cv2.imwrite(image_filename, cropped)
    print(f"Saved image: {image_filename}")

    # Draw "Saved" text inside the bounding box on the main frame
    if bbox:
        x1, y1, _, _ = bbox
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        border_thickness = 5
        text = "Saved"
        # Ensure text position is valid within frame dimensions
        h, w = last_frame.shape[:2]
        # Basic check to keep text position roughly within bounds
        text_x = min(max(x1 + 10, 0), w - 100)
        text_y = min(max(y1 + 30, 0), h - 10)

        text_position = (text_x, text_y)

        # Black border
        cv2.putText(last_frame, text, text_position, font, font_scale,
                    (0, 0, 0), border_thickness, cv2.LINE_AA)
        # White text
        cv2.putText(last_frame, text, text_position, font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)

        # Update the main window to show the "Saved" text
        draw_main_window()

    # Append angles to CSV
    if all(a is not None for a in angles):
        try:
            with open(output_csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([idx] + [f"{a:.2f}" for a in angles])
            print(f"Appended angles to: {output_csv_path}")
        except Exception as e:
            print(f"Error writing to CSV {output_csv_path}: {e}")
    else:
        print("Angles not computed, skipping CSV entry.")

def main():
    global paused, detection_needed, last_frame, idx, total, bbox, points, video_basename, output_dir, output_csv_path
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", default="yolo11l-pose.pt",
                        choices=["yolo11n-pose.pt","yolo11s-pose.pt",
                                 "yolo11m-pose.pt","yolo11l-pose.pt",
                                 "yolo11x-pose.pt"])
    parser.add_argument("--output", default="output", help="Base directory for output files")
    args = parser.parse_args()

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise IOError(f"Cannot open {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = 0
    paused = False
    detection_needed = False

    video_basename = os.path.splitext(os.path.basename(args.video))[0]

    # Create timestamped output directory
    output_base_dir = args.output
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base_dir, f"{video_basename}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize CSV file
    output_csv_path = os.path.join(output_dir, video_basename + ".csv")
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Frame", "S-H-K", "H-K-A", "K-A-T"])
    print(f"Saving results to: {output_dir}")

    cv2.namedWindow(video_basename, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(video_basename, mouse_cb)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret or idx >= total:
                break
            last_frame = frame.copy()
            idx += 1
            draw_main_window()
        else:
            if detection_needed:
                det = run_detection(last_frame, model)
                if det:
                    bbox, points = det
                    recompute_angles()
                    draw_main_window()
                detection_needed = False

        key = cv2.waitKeyEx(DELAY_MS if not paused else 0)
        if key == 32:  # space
            paused = not paused
            if paused:
                detection_needed = True
            else:
                bbox = None
                points = []
        elif key == 2555904 or key == ord('n'): # right arrow or 'n'
            paused = True
            idx = min(idx + 1, total - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            last_frame = frame.copy()
            detection_needed = True
        elif key == 2424832 or key == ord('p'): # left arrow or 'p'
            paused = True
            idx = max(idx - 1, 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            last_frame = frame.copy()
            detection_needed = True
        elif key == ord('s') and paused:
            save_cropped_image()
        elif key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
