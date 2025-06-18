# Real-time head posture detection
Simple project demonstrating mediapipe and opencv for real-time head posture detection.

# How It Works
- Captures video from webcam
- Uses MediaPipe to detect body landmarks
- Calculates the 3D angle formed between the shoulder, ear, and nose
- If the angle exceeds 30°, a warning is displayed on screen



## Requirements

- Python 3.7 or higher
- Webcam

## Installation and Running

```bash
sudo apt install python3 python3-pip python3-venv # Example of python installation on linux. See https://www.python.org/downloads/

git clone https://github.com/henriqueluhm/head-posture-detection.git
cd head-posture-detection

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

python3 main.py # Run the script
```

Press "q" to quit the application.

## Note
The script uses a system font at /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf for Unicode text rendering, based on Debian. 

If you're on another OS, update the `font_path` variable in the script:

- **Windows:** C:\Windows\Fonts\arial.ttf
- **macOS:** /System/Library/Fonts/Supplemental/Arial.ttf

## Troubleshooting
Black screen or no landmarks? Make sure your webcam is not in use by another app.

Text not showing? Check if the font path in font_path is valid on your OS.

## Dependencies
List of dependencies is in requirements.txt. To regenerate it:

```bash
pip freeze > requirements.txt
```

## Licence
MIT