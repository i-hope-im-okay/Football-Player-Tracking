# Football-Player-Tracking
This is a ptoject that uses computer vision, and video processing using python. takes a video of a socker match and returns an .mp4 file which highlights players and theor jersey numbers using colored boxes throughout the video

## Overview
This is a project that uses YOLOv8 and DeepSORT in python to track soccer players in a video clip, annotating them with green bounding boxes and simulated jersey numbers.

## Files
- `player_tracking.py`: Main script for player detection and tracking.
- `output.mp4`: Processed video with tracking annotations.
- `demo.jpg`: Screenshot of tracking results.

## Dependencies
- Python 3.8+
- ultralytics
- deep-sort-realtime==1.3.2
- opencv-python
- torch

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Place `football_clip.mp4` (https://drive.google.com/file/d/1tXA_CmcSh20gDpfgEyW-BnIKoLL3nDAV/view?usp=drive_link) in the same directory.
4. Run: `python player_tracking.py`

## Demo
![Tracking Demo](demo.jpg)
[Output Video](https://drive.google.com/file/d/1ZI_cex4RXzD8ezsXfRSyoWWK5GUDaLD_/view?usp=drive_link) <!-- Replace with Google Drive link if >25MB, e.g., https://drive.google.com/your-link -->

## Author
Aarav Sharma
