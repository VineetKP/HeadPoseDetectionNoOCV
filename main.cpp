// head_pose_main.cpp
//
// Head pose estimation — OpenCV used only for I/O / face detection.
// All pose math comes from pose_math.hpp.
//
// FIXES applied (cumulative):
//  1. fx == fy == frame.cols            (square pixels)
//  2. 3D model Y-axis flipped           (Y-down = OpenCV camera convention)
//  3. Numerical Jacobian in LM solver   (eliminates wild fluctuations)
//  4. Correct ZYX Euler extraction
//  5. Calibration phase                 (eliminates the residual constant offset
//                                        caused by the generic 3D model not
//                                        matching the user's actual face geometry)
//
// HOW TO USE CALIBRATION:
//   When the program starts it shows "CALIBRATION" on screen.
//   Look straight at the camera for ~2 seconds (CALIB_FRAMES frames).
//   After that all angles are reported relative to your neutral pose,
//   so pitch=yaw=roll=0 means "looking straight at the camera".

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include "pose.hpp"

using namespace cv;
using namespace cv::face;
using namespace std;

// Number of frames collected for calibration baseline.
// At 15 fps this is ~2 seconds. Increase if you want a longer calibration.
static constexpr int CALIB_FRAMES = 30;

int main() {
    // ----------------------------------------------------------------
    // Load detector + landmark model (OpenCV I/O only)
    // ----------------------------------------------------------------
    CascadeClassifier faceDetector;
    if (!faceDetector.load("../haarcascade_frontalface_default.xml")) {
        cerr << "Error loading face detector!" << endl; return -1;
    }
    Ptr<Facemark> facemark = FacemarkLBF::create();
    facemark->loadModel("../lbfmodel.yaml");

    // VideoCapture cap("../2026-03-13-234738.webm");
    VideoCapture cap(0);

    if (!cap.isOpened()) {
        cerr << "Error opening video file!" << endl; return -1;
    }

    // ----------------------------------------------------------------
    // 3D face model — Y-DOWN convention to match OpenCV camera coords.
    //
    // Landmark indices (68-pt LBF):
    //   30 = nose tip   (origin)
    //    8 = chin        Y+ (below nose in image)
    //   36 = left eye    Y- (above nose in image)
    //   45 = right eye   Y-
    //   48 = left mouth  Y+
    //   54 = right mouth Y+
    // ----------------------------------------------------------------
    vector<pm::Point3d> objPt = {
        {  0.0,    0.0,    0.0 },   // 30 nose tip
        {  0.0,   63.6,  -12.5 },   //  8 chin
        {-43.3,  -32.7,  -26.0 },   // 36 left eye outer corner
        { 43.3,  -32.7,  -26.0 },   // 45 right eye outer corner
        {-28.9,   28.9,  -24.1 },   // 48 left mouth corner
        { 28.9,   28.9,  -24.1 }    // 54 right mouth corner
    };

    // ----------------------------------------------------------------
    // Calibration state
    // ----------------------------------------------------------------
    vector<double> cal_p, cal_y, cal_r;
    double off_p = 0, off_y = 0, off_r = 0;
    bool calibrated = false;

    cout << "=== CALIBRATION ===" << endl;
    cout << "Look straight at the camera. Collecting "
         << CALIB_FRAMES << " frames..." << endl;

    // ----------------------------------------------------------------
    // Main loop
    // ----------------------------------------------------------------
    Mat frame;
    while (cap.read(frame)) {
        vector<Rect> faces;
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        faceDetector.detectMultiScale(gray, faces);

        // Show calibration overlay
        if (!calibrated) {
            string msg = "CALIBRATION: look straight  ["
                         + to_string((int)cal_p.size())
                         + "/" + to_string(CALIB_FRAMES) + "]";
            putText(frame, msg, {10, 35}, FONT_HERSHEY_SIMPLEX,
                    0.7, {0, 255, 255}, 2);
        }

        for (const auto& face : faces) {
            rectangle(frame, face, Scalar(255, 0, 0), 2);

            vector<vector<Point2f>> landmarks;
            if (!facemark->fit(frame, faces, landmarks)) continue;

            // Camera intrinsics — square pixels, approximate focal length
            double fx = frame.cols, fy = frame.cols;
            double cx = frame.cols / 2.0, cy = frame.rows / 2.0;

            // 2D landmark positions
            vector<pm::Point2d> imgPt = {
                { landmarks[0][30].x, landmarks[0][30].y },
                { landmarks[0][8 ].x, landmarks[0][8 ].y },
                { landmarks[0][36].x, landmarks[0][36].y },
                { landmarks[0][45].x, landmarks[0][45].y },
                { landmarks[0][48].x, landmarks[0][48].y },
                { landmarks[0][54].x, landmarks[0][54].y }
            };

            // Solve for camera pose
            pm::Vec3 rvec, tvec;
            if (!pm::solvePnP(objPt, imgPt, fx, fy, cx, cy, rvec, tvec)) continue;

            pm::Mat33 R = pm::rodrigues(rvec);
            if (pm::checkOrthonormal(R) >= 1e-4) continue;

            double pitch, yaw, roll;
            pm::eulerAngles(R, pitch, yaw, roll);
            const double toDeg = 180.0 / M_PI;
            pitch *= toDeg; yaw *= toDeg; roll *= toDeg;

            // ---- Calibration phase: collect frames ----
            if (!calibrated) {
                cal_p.push_back(pitch);
                cal_y.push_back(yaw);
                cal_r.push_back(roll);

                if ((int)cal_p.size() >= CALIB_FRAMES) {
                    // Use median (robust to outlier frames during calibration)
                    auto median = [](vector<double> v) -> double {
                        sort(v.begin(), v.end());
                        int n = (int)v.size();
                        return (n % 2 == 0)
                               ? (v[n/2-1] + v[n/2]) * 0.5
                               : v[n/2];
                    };
                    off_p = median(cal_p);
                    off_y = median(cal_y);
                    off_r = median(cal_r);
                    calibrated = true;
                    cout << "Calibration complete." << endl;
                    cout << "  Pitch offset: " << off_p << " deg" << endl;
                    cout << "  Yaw offset:   " << off_y << " deg" << endl;
                    cout << "  Roll offset:  " << off_r << " deg" << endl;
                    cout << "===========================" << endl;
                }
                // Don't print angles while calibrating
                continue;
            }

            // ---- Runtime: subtract calibration offsets ----
            double p = pitch - off_p;
            double y = yaw   - off_y;
            double r = roll  - off_r;

            cout << "Pitch (nod):  " << p << " deg" << endl;
            cout << "Yaw   (turn): " << y << " deg" << endl;
            cout << "Roll  (tilt): " << r << " deg" << endl;
            cout << "---" << endl;

            // Draw angles on frame
            auto fmtDeg = [](double v) {
                char buf[24];
                snprintf(buf, sizeof(buf), "%+.1f deg", v);
                return string(buf);
            };
            putText(frame, "Pitch: " + fmtDeg(p), {10,  60},
                    FONT_HERSHEY_SIMPLEX, 0.65, {0,255,0}, 2);
            putText(frame, "Yaw:   " + fmtDeg(y), {10,  90},
                    FONT_HERSHEY_SIMPLEX, 0.65, {0,255,0}, 2);
            putText(frame, "Roll:  " + fmtDeg(r), {10, 120},
                    FONT_HERSHEY_SIMPLEX, 0.65, {0,255,0}, 2);

            // Draw pose axes (X=red Y=green Z=blue) from nose tip
            vector<pm::Point3d> axisW = {{100,0,0},{0,100,0},{0,0,-100}};
            auto axisI = pm::projectPoints(axisW, rvec, tvec, fx, fy, cx, cy);
            cv::Point nose((int)landmarks[0][30].x, (int)landmarks[0][30].y);
            line(frame, nose, {(int)axisI[0].x,(int)axisI[0].y}, {0,  0,255}, 2);
            line(frame, nose, {(int)axisI[1].x,(int)axisI[1].y}, {0,255,  0}, 2);
            line(frame, nose, {(int)axisI[2].x,(int)axisI[2].y}, {255,0,  0}, 2);
        }

        imshow("Head Pose", frame);
        if (waitKey(10) == 27) break;  // ESC
    }

    cap.release();
    destroyAllWindows();
    return 0;
}