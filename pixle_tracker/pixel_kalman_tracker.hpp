#pragma once

#include <array>

/**
 * @brief Tracks a 2D pixel point (u, v) using a Kalman Filter.
 *
 * State vector: [u, v, du, dv]  (position + velocity)
 * Measurement:  [u, v]
 *
 * All matrix math is hand-rolled — no OpenCV, no Eigen.
 *
 * @param u             Measured pixel column (x-axis)
 * @param v             Measured pixel row    (y-axis)
 * @param est_u         [out] Filtered estimate of u
 * @param est_v         [out] Filtered estimate of v
 * @param reset         Pass true on the very first call (or to reinitialise)
 * @param dt            Time step in seconds (default 1.0 for frame-to-frame)
 * @param proc_noise_q  Process  noise scalar Q (tune: higher = trust motion more)
 * @param meas_noise_r  Measurement noise scalar R (tune: higher = trust sensor less)
 */
inline void trackPixelKalman(
    double  u,
    double  v,
    double& est_u,
    double& est_v,
    bool    reset         = false,
    double  dt            = 1.0,
    double  proc_noise_q  = 1e-2,
    double  meas_noise_r  = 1e+1)
{
    // -----------------------------------------------------------------------
    // Persistent state (survives across calls via static storage)
    // -----------------------------------------------------------------------
    // State:            x = [u, v, du, dv]^T          (4 x 1)
    // Covariance:       P                              (4 x 4)
    // -----------------------------------------------------------------------
    static double x[4];          // state vector
    static double P[4][4];       // error covariance
    static bool   initialised = false;

    // -----------------------------------------------------------------------
    // Tiny local lambdas (as nested helpers via auto — C++14)
    // -----------------------------------------------------------------------

    // 4x4 matrix multiply:  C = A * B
    auto mat4x4_mul = [](const double A[4][4],
                         const double B[4][4],
                         double       C[4][4])
    {
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) {
                C[i][j] = 0.0;
                for (int k = 0; k < 4; ++k)
                    C[i][j] += A[i][k] * B[k][j];
            }
    };

    // 4x4 + 4x4 matrix add:  C = A + B
    auto mat4x4_add = [](const double A[4][4],
                         const double B[4][4],
                         double       C[4][4])
    {
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                C[i][j] = A[i][j] + B[i][j];
    };

    // Transpose of a 4x4 matrix:  B = A^T
    auto mat4x4_T = [](const double A[4][4], double B[4][4])
    {
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                B[i][j] = A[j][i];
    };

    // Invert a 2x2 matrix (used for innovation covariance S)
    // Returns false if singular
    auto mat2x2_inv = [](const double M[2][2], double inv[2][2]) -> bool
    {
        double det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
        if (det < 1e-12 && det > -1e-12) return false;
        double inv_det = 1.0 / det;
        inv[0][0] =  M[1][1] * inv_det;
        inv[0][1] = -M[0][1] * inv_det;
        inv[1][0] = -M[1][0] * inv_det;
        inv[1][1] =  M[0][0] * inv_det;
        return true;
    };

    // -----------------------------------------------------------------------
    // Initialise / reset
    // -----------------------------------------------------------------------
    if (!initialised || reset) {
        x[0] = u;   // initial u
        x[1] = v;   // initial v
        x[2] = 0.0; // initial du (velocity)
        x[3] = 0.0; // initial dv

        // Identity covariance with large uncertainty
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                P[i][j] = (i == j) ? 1e3 : 0.0;

        initialised = true;
        est_u = u;
        est_v = v;
        return;
    }

    // -----------------------------------------------------------------------
    // Kalman Filter — Constant Velocity Model
    // -----------------------------------------------------------------------

    // ------ 1. Build state-transition matrix F (4x4) ----------------------
    //
    //   | 1  0  dt  0 |
    //   | 0  1   0 dt |
    //   | 0  0   1  0 |
    //   | 0  0   0  1 |
    //
    double F[4][4] = {
        {1, 0, dt,  0},
        {0, 1,  0, dt},
        {0, 0,  1,  0},
        {0, 0,  0,  1}
    };

    // ------ 2. Build process noise matrix Q (4x4) -------------------------
    //
    //  Discrete-time constant-velocity noise (assumes acceleration is noise):
    //
    //   Q = q * | dt^4/4  0       dt^3/2  0      |
    //           | 0       dt^4/4  0       dt^3/2 |
    //           | dt^3/2  0       dt^2    0      |
    //           | 0       dt^3/2  0       dt^2   |
    //
    double dt2 = dt  * dt;
    double dt3 = dt2 * dt;
    double dt4 = dt3 * dt;

    double Q[4][4] = {
        {proc_noise_q * dt4 / 4.0,  0,                          proc_noise_q * dt3 / 2.0,  0                        },
        {0,                          proc_noise_q * dt4 / 4.0,  0,                          proc_noise_q * dt3 / 2.0},
        {proc_noise_q * dt3 / 2.0,  0,                          proc_noise_q * dt2,         0                        },
        {0,                          proc_noise_q * dt3 / 2.0,  0,                          proc_noise_q * dt2       }
    };

    // ------ 3. Measurement matrix H (2x4) ----------------------------------
    //
    //   H = | 1  0  0  0 |
    //       | 0  1  0  0 |
    //
    //   (we only observe u and v, not velocities)
    //
    double H[2][4] = {
        {1, 0, 0, 0},
        {0, 1, 0, 0}
    };

    // ------ 4. Measurement noise matrix R (2x2) ----------------------------
    double R[2][2] = {
        {meas_noise_r, 0            },
        {0,            meas_noise_r }
    };

    // =========================================================
    // PREDICT
    // =========================================================

    // x_pred = F * x   (4x1)
    double x_pred[4] = {0, 0, 0, 0};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            x_pred[i] += F[i][j] * x[j];

    // P_pred = F * P * F^T + Q   (4x4)
    double FP[4][4], FT[4][4], FPFT[4][4], P_pred[4][4];
    mat4x4_mul(F, P, FP);
    mat4x4_T(F, FT);
    mat4x4_mul(FP, FT, FPFT);
    mat4x4_add(FPFT, Q, P_pred);

    // =========================================================
    // UPDATE
    // =========================================================

    // Innovation:  y = z - H * x_pred   (2x1)
    // z = [u, v]^T
    double z[2] = {u, v};
    double Hx[2] = {0, 0};
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 4; ++j)
            Hx[i] += H[i][j] * x_pred[j];

    double innov[2] = {z[0] - Hx[0], z[1] - Hx[1]};

    // Innovation covariance:  S = H * P_pred * H^T + R   (2x2)
    // Compute H * P_pred  (2x4)
    double HP[2][4] = {};
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 4; ++k)
                HP[i][j] += H[i][k] * P_pred[k][j];

    // H^T  (4x2)
    double HT[4][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 4; ++j)
            HT[j][i] = H[i][j];

    // HP * H^T  (2x2)
    double HPHT[2][2] = {};
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 4; ++k)
                HPHT[i][j] += HP[i][k] * HT[k][j];

    double S[2][2] = {
        {HPHT[0][0] + R[0][0],  HPHT[0][1] + R[0][1]},
        {HPHT[1][0] + R[1][0],  HPHT[1][1] + R[1][1]}
    };

    // S^{-1}  (2x2)
    double S_inv[2][2];
    if (!mat2x2_inv(S, S_inv)) {
        // Singular — skip update, carry prediction forward
        for (int i = 0; i < 4; ++i) x[i] = x_pred[i];
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                P[i][j] = P_pred[i][j];
        est_u = x[0];
        est_v = x[1];
        return;
    }

    // Kalman Gain:  K = P_pred * H^T * S^{-1}   (4x2)
    // P_pred * H^T  (4x2)
    double PHT[4][2] = {};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 4; ++k)
                PHT[i][j] += P_pred[i][k] * HT[k][j];

    // K = PHT * S_inv  (4x2)
    double K[4][2] = {};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k)
                K[i][j] += PHT[i][k] * S_inv[k][j];

    // Updated state:  x = x_pred + K * innov   (4x1)
    for (int i = 0; i < 4; ++i) {
        x[i] = x_pred[i];
        for (int j = 0; j < 2; ++j)
            x[i] += K[i][j] * innov[j];
    }

    // Updated covariance:  P = (I - K * H) * P_pred   (4x4)
    // K * H  (4x4)
    double KH[4][4] = {};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 2; ++k)
                KH[i][j] += K[i][k] * H[k][j];

    // I - K*H
    double I_KH[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            I_KH[i][j] = (i == j ? 1.0 : 0.0) - KH[i][j];

    // P = (I - K*H) * P_pred
    double P_new[4][4];
    mat4x4_mul(I_KH, P_pred, P_new);

    // Copy back
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            P[i][j] = P_new[i][j];

    // -----------------------------------------------------------------------
    // Output filtered position
    // -----------------------------------------------------------------------
    est_u = x[0];
    est_v = x[1];
}