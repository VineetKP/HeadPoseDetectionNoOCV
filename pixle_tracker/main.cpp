// kalman_nose_tracker.cpp
//
// Based on head_pose_main.cpp — face detection and landmarks are identical.
// Head pose math removed. Nose tip (landmark 30) is fed into
// trackPixelKalman() and the result is visualised live.
//
// Keys:  ESC / Q  — quit       R — reset tracker

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <deque>
#include <cmath>
#include <algorithm>
#include <string>
#include "pixel_kalman_tracker.hpp"

using namespace cv;
using namespace cv::face;
using namespace std;

// ─────────────────────────────────────────────────────────────────────────────
//  Tuning
// ─────────────────────────────────────────────────────────────────────────────
static const int    TRAIL_LEN    = 50;
static const int    NOSE_IDX     = 30;
static const double DT           = 1.0 / 30.0;
static const double PROC_NOISE_Q = 5e-2;
static const double MEAS_NOISE_R = 8.0;

// BGR colours
static const Scalar C_LM  (180, 180, 180);
static const Scalar C_BOX ( 60, 200,  60);
static const Scalar C_RAW ( 40, 100, 255);   // orange-red  – raw measurement
static const Scalar C_KAL ( 40, 230, 120);   // green       – Kalman estimate
static const Scalar C_TRM ( 20,  60, 160);   // trail raw
static const Scalar C_TKL ( 20, 150,  80);   // trail kalman
static const Scalar C_TXT (240, 240, 240);
static const Scalar C_BLK (  0,   0,   0);

// ─────────────────────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────────────────────
static void overlayRect(Mat& img, Rect r, Scalar col, double a)
{
    r &= Rect(0, 0, img.cols, img.rows);
    if (r.empty()) return;
    Mat roi  = img(r);
    Mat fill(roi.size(), roi.type(), col);
    addWeighted(fill, a, roi, 1.0 - a, 0, roi);
}

static void drawHUD(Mat& f,
                    double ru, double rv,
                    double ku, double kv, int fps)
{
    const int P=10, LH=22, W=310, H=138;
    overlayRect(f, {P, P, W, H}, C_BLK, 0.55);

    auto put = [&](const string& s, int row, Scalar c, double sc=0.50)
    {
        putText(f, s, {P*2, P+LH+row*LH},
                FONT_HERSHEY_SIMPLEX, sc, c, 1, LINE_AA);
    };

    put("NOSE TRACKER  [Kalman]", 0, C_TXT, 0.54);

    circle(f, {P*2+4, P+LH*2+6}, 5, C_RAW, -1, LINE_AA);
    put(" Raw    u:"+to_string((int)ru)+"  v:"+to_string((int)rv), 2, C_RAW);

    circle(f, {P*2+4, P+LH*3+6}, 5, C_KAL, -1, LINE_AA);
    put(" Kalman u:"+to_string((int)ku)+"  v:"+to_string((int)kv), 3, C_KAL);

    double d = sqrt((ru-ku)*(ru-ku)+(rv-kv)*(rv-kv));
    string ds = to_string(d); ds = ds.substr(0, ds.find('.')+3);
    put("Innovation: "+ds+" px", 4, C_TXT, 0.46);
    put("FPS:"+to_string(fps)+"  [R]=reset  [Q/ESC]=quit", 5, C_TXT, 0.44);
}

static void drawTrail(Mat& f,
                      const deque<Point2d>& tr,
                      Scalar col, int base=2)
{
    int n = (int)tr.size();
    for (int i = 1; i < n; ++i) {
        double a = (double)i / n;
        line(f,
             {(int)tr[i-1].x, (int)tr[i-1].y},
             {(int)tr[i  ].x, (int)tr[i  ].y},
             col * a, max(1, (int)(base*a)), LINE_AA);
    }
}

static void drawArrow(Mat& f, Point2d pos, Point2d prev, double sc=10.0)
{
    Point2d v = (pos - prev) * sc;
    if (norm(v) < 1.5) return;
    arrowedLine(f,
                {(int)pos.x,        (int)pos.y},
                {(int)(pos.x+v.x),  (int)(pos.y+v.y)},
                C_KAL, 2, LINE_AA, 0, 0.30);
}

// Draw 68-pt facial mesh (same connectivity as original)
static void drawLandmarks(Mat& f,
                          const vector<Point2f>& lm,
                          Rect box)
{
    rectangle(f, box, C_BOX, 1, LINE_AA);

    auto poly = [&](int a, int b, bool closed=false) {
        vector<Point> pts;
        for (int i = a; i <= b; ++i)
            pts.push_back({(int)lm[i].x, (int)lm[i].y});
        polylines(f, pts, closed, C_LM, 1, LINE_AA);
    };

    poly(0,  16);        // jaw
    poly(17, 21);        // left brow
    poly(22, 26);        // right brow
    poly(27, 30);        // nose bridge
    poly(30, 35, true);  // nose base
    poly(36, 41, true);  // left eye
    poly(42, 47, true);  // right eye
    poly(48, 59, true);  // outer lip
    poly(60, 67, true);  // inner lip

    for (int i = 0; i < (int)lm.size(); ++i) {
        bool isNose = (i == NOSE_IDX);
        circle(f, {(int)lm[i].x, (int)lm[i].y},
               isNose ? 5 : 2,
               isNose ? C_KAL : C_LM,
               -1, LINE_AA);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────
int main()
{
    // ── Same setup as head_pose_main.cpp ─────────────────────────────────
    CascadeClassifier faceDetector;
    if (!faceDetector.load("../haarcascade_frontalface_default.xml")) {
        cerr << "Error loading face detector!\n"; return -1;
    }

    Ptr<Facemark> facemark = FacemarkLBF::create();
    facemark->loadModel("../lbfmodel.yaml");

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error opening camera!\n"; return -1;
    }

    // ── Tracker state ─────────────────────────────────────────────────────
    deque<Point2d> trail_raw, trail_kal;
    Point2d prev_kal{0, 0};
    bool first_frame = true;

    int64  tick0 = getTickCount();
    int    fps   = 0;

    cout << "Kalman Nose Tracker running.\n"
         << "  ESC / Q — quit\n"
         << "  R       — reset tracker\n";

    Mat frame;
    while (cap.read(frame))
    {
        Mat display = frame.clone();

        // Rolling FPS (updated every 0.5 s)
        int64  tnow   = getTickCount();
        double elapsed = (tnow - tick0) / getTickFrequency();
        if (elapsed >= 0.5) {
            fps   = (int)round(1.0 / elapsed);
            tick0 = tnow;
        }

        // ── Face detection (identical to original) ────────────────────────
        vector<Rect> faces;
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        faceDetector.detectMultiScale(gray, faces);

        if (faces.empty()) {
            putText(display, "No face detected",
                    {20, display.rows - 24},
                    FONT_HERSHEY_SIMPLEX, 0.65, {0,80,255}, 1, LINE_AA);
            imshow("Kalman Nose Tracker", display);
            int k = waitKey(10) & 0xFF;
            if (k == 27 || k == 'q') break;
            continue;
        }

        // ── Landmark detection (identical to original) ────────────────────
        vector<vector<Point2f>> landmarks;
        if (!facemark->fit(frame, faces, landmarks) || landmarks.empty()) {
            imshow("Kalman Nose Tracker", display);
            if ((waitKey(10) & 0xFF) == 27) break;
            continue;
        }

        // Use first face only (same as original)
        const auto& lm = landmarks[0];
        drawLandmarks(display, lm, faces[0]);

        // ── Feed nose tip into Kalman filter ──────────────────────────────
        double raw_u = lm[NOSE_IDX].x;
        double raw_v = lm[NOSE_IDX].y;

        double kal_u, kal_v;
        trackPixelKalman(raw_u, raw_v,
                         kal_u, kal_v,
                         first_frame,
                         DT, PROC_NOISE_Q, MEAS_NOISE_R);

        if (first_frame) {
            prev_kal   = {kal_u, kal_v};
            first_frame = false;
        }

        // Update trails
        trail_raw.push_back({raw_u, raw_v});
        trail_kal.push_back({kal_u, kal_v});
        if ((int)trail_raw.size() > TRAIL_LEN) trail_raw.pop_front();
        if ((int)trail_kal.size() > TRAIL_LEN) trail_kal.pop_front();

        // ── Visualise ─────────────────────────────────────────────────────
        drawTrail(display, trail_raw, C_TRM, 2);
        drawTrail(display, trail_kal, C_TKL, 2);

        drawArrow(display, {kal_u, kal_v}, prev_kal, 10.0);
        prev_kal = {kal_u, kal_v};

        // Raw measurement dot (orange-red)
        circle(display, {(int)raw_u, (int)raw_v}, 6, C_RAW, -1, LINE_AA);
        circle(display, {(int)raw_u, (int)raw_v}, 9, C_RAW,  1, LINE_AA);

        // Kalman estimate dot (green, larger ring)
        circle(display, {(int)kal_u, (int)kal_v},  6, C_KAL, -1, LINE_AA);
        circle(display, {(int)kal_u, (int)kal_v}, 11, C_KAL,  2, LINE_AA);

        // Connector line between raw and filtered
        line(display,
             {(int)raw_u, (int)raw_v},
             {(int)kal_u, (int)kal_v},
             {160,160,160}, 1, LINE_AA);

        // Labels
        putText(display, "RAW",
                {(int)raw_u+11, (int)raw_v-8},
                FONT_HERSHEY_SIMPLEX, 0.42, C_RAW, 1, LINE_AA);
        putText(display, "KALMAN",
                {(int)kal_u+11, (int)kal_v+16},
                FONT_HERSHEY_SIMPLEX, 0.42, C_KAL, 1, LINE_AA);

        drawHUD(display, raw_u, raw_v, kal_u, kal_v, fps);

        imshow("Kalman Nose Tracker", display);
        int key = waitKey(10) & 0xFF;
        if (key == 27 || key == 'q') break;
        if (key == 'r') {
            first_frame = true;
            trail_raw.clear();
            trail_kal.clear();
            cout << "[INFO] Tracker reset\n";
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}