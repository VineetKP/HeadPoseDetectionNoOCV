# Face Landmark Detection

This project detects faces in a video stream and overlays 68-point facial landmarks using OpenCV.

# HeadPose Detection

Once landmarks are detected, 6 facial points coordinates(world coordinates and image coordinates) are passed to the nonOpenCV based implementation to identify the rotation matrix. Final head pose rotation is derived from the same.

## Requirements

- OpenCV
- Pre-trained models:
  - `haarcascade_frontalface_default.xml`
  - `lbfmodel.yaml`

## Build Instructions

1. Install OpenCV on your system.
2. Place the required pre-trained models in the project directory.
3. Build the project:

```bash
mkdir build
cd build
cmake ..
make
```

4. Run the executable:

```bash
./FaceLandmarkDetection
```

## Usage

- The program uses the default camera (change `VideoCapture` source in `main.cpp` if needed).
- Press `ESC` to exit the program.

## Notes

- Ensure the pre-trained models are in the same directory as the executable.
- Adjust the camera index in `VideoCapture` if using an external camera.