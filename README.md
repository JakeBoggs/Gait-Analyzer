# Gait Analyzer Tool

This tool analyzes gait patterns from side-view videos of walking patients using YOLOv11-pose estimation. It allows users to interactively review video frames, adjust key joint points, calculate relevant angles, and save the results. Once the models are downloaded, the app runs entirely on-device without requiring network access, to ensure HIPAA compliance. Created over an Easter weekend for my mother, who needed to analyze a dataset of videos for her research study.

## Demo

https://github.com/user-attachments/assets/8a7494c0-91d0-483e-bdd4-6f00227b30ea

## Features

*   **Video Playback:** Play, pause (Space), and step through video frames (Left/Right Arrows or 'p'/'n').
*   **Pose Estimation:** Automatically detects human poses using YOLOv11-pose when paused or stepping.
*   **5-Point Model:** Focuses on key lower body points:
    *   **S:** Shoulder Center (midpoint between left/right shoulders)
    *   **H:** Hip Center (midpoint between left/right hips)
    *   **K:** Knee (left or right, whichever has higher confidence)
    *   **A:** Ankle (corresponding to the selected knee)
    *   **T:** Toe (calculated perpendicular to the ankle-knee line, 0.5 * ||KA|| length)
*   **Angle Calculation:** Computes three angles critical for gait analysis:
    *   **S-H-K:** Angle between Shoulder Center, Hip Center, and Knee.
    *   **H-K-A:** Angle between Hip Center, Knee, and Ankle.
    *   **K-A-T:** Angle between Knee, Ankle, and calculated Toe point.
*   **Interactive Adjustment:** Drag and drop the 5 key points (S, H, K, A, T) to refine their positions when paused. Angles are recalculated automatically.
*   **Save Results:** Press 's' (when paused) to save:
    *   A cropped image of the detected person with points and angles overlaid.
    *   A CSV file containing the frame number and the three calculated angles for each saved frame.

## Requirements

*   Python 3 ([Download here](https://www.python.org/downloads/))
*   OpenCV (`opencv-python`)
*   NumPy (`numpy`)
*   Ultralytics YOLO (`ultralytics`)

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/JakeBoggs/Gait-Analyzer.git
    cd Gait-Analyzer
    ```
2.  Install the required dependencies:
    ```bash
    pip install opencv-python numpy ultralytics
    ```
    The YOLOv11 pose models will be downloaded automatically on first use if they are not found locally.

## Usage

Run the script from the command line:

```bash
python track.py --video <path_to_video_file> [--model <model_name.pt>] [--output <output_directory>]
```

**Arguments:**

*   `--video` (required): Path to the input video file (ideally a side-view recording of walking).
*   `--model` (optional): Name of the YOLOv11-pose model file to use. Defaults to `yolov11l-pose.pt`. Choices include `yolov11n-pose.pt`, `yolov11s-pose.pt`, `yolov11m-pose.pt`, `yolov11l-pose.pt`, `yolov11x-pose.pt`.
*   `--output` (optional): Base directory to save output files. Defaults to `output`.

**Example:**

```bash
python track.py --video my_gait_video.mp4 --model yolov11l-pose.pt --output output_dir
```

## Keyboard Controls

*   **Space:** Pause/Resume video playback. Detection runs when paused.
*   **Right Arrow / 'n':** Step forward one frame (pauses video).
*   **Left Arrow / 'p':** Step backward one frame (pauses video).
*   **'s':** Save the current frame's cropped image and angles (only when paused and detection has run).
*   **Esc:** Quit the application.

## Output

The script creates a timestamped subdirectory within the specified output directory (or the default `output` directory). For example: `output/my_gait_video_20231027_103000/`

Inside this directory, you will find:

*   **Cropped Images:** PNG images named `<video_basename>_<frame_number>.png` for each saved frame (e.g., `my_gait_video_0123.png`). These images show the detected bounding box area with the 5 points and calculated angles overlaid.
*   **CSV File:** A CSV file named `<video_basename>.csv` (e.g., `my_gait_video.csv`) containing the frame number and the three calculated angles (S-H-K, H-K-A, K-A-T) for each saved frame.

## Point Visualization Colors

*   **S (Shoulder Center):** Green
*   **H (Hip Center):** Blue
*   **K (Knee):** Red
*   **A (Ankle):** Yellow
*   **T (Toe):** Cyan 
