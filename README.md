# Blink to Click - Face & Eye Tracking Mouse Control

## Overview
This project enables hands-free mouse control using facial tracking and blink detection. It uses OpenCV, MediaPipe, and PyAutoGUI to track facial landmarks, detect blinks, and move the cursor based on nose position.

## Features
- **Face & Eye Tracking:** Uses MediaPipe Face Mesh for landmark detection.
- **Blink Detection:** Detects blinks to trigger mouse clicks.
- **Cursor Control:** Moves the cursor based on nose position.
- **CUDA Acceleration:** Uses GPU if available for faster processing.

## Requirements
- Python 3.x
- OpenCV
- MediaPipe
- PyAutoGUI
- NumPy

### Install Dependencies
```sh
pip install opencv-python mediapipe pyautogui numpy
```

## Usage
1. Run the script:
   ```sh
   python blink_mouse_control.py
   ```
2. The program will start tracking your face.
3. Move your head to control the cursor.
4. Blink to simulate a mouse click.
5. Press `q` to exit.

## Configuration
- Adjust `BLINK_THRESHOLD`, `MOVE_X`, and `MOVE_Y` to fine-tune the behavior.
- Uses CUDA if available for improved performance.

## License
This project is open-source. Feel free to modify and improve!

