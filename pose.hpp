#pragma once
// ============================================================
//  pose_math.hpp  —  OpenCV-independent head pose estimation
//
//  Provides exact numerical equivalents of:
//    cv::solvePnP()      — DLT init + Levenberg-Marquardt refinement
//    cv::Rodrigues()     — rvec <-> R matrix
//    cv::projectPoints() — 3D -> 2D projection
//    Euler angle extraction (ZYX convention)
//
//  Stability guarantees vs the previous version:
//  - Numerical Jacobian in LM loop  (avoids the broken analytic
//    Rodrigues derivative that caused the wild fluctuations)
//  - Full Levenberg-Marquardt with adaptive damping (not plain GN)
//  - SVD-based DLT with SO(3) projection for initialisation
// ============================================================

#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>

namespace pm {

// ------------------------------------------------------------------
// Basic point types
// ------------------------------------------------------------------
struct Point2d { double x, y; };
struct Point3d { double x, y, z; };

// 3x3 rotation matrix
struct Mat33 {
    double d[3][3] = {};
    double& at(int r,int c)       { return d[r][c]; }
    double  at(int r,int c) const { return d[r][c]; }
};

// 3-vector (used for rvec and tvec)
struct Vec3 {
    double d[3] = {};
    double& operator[](int i)       { return d[i]; }
    double  operator[](int i) const { return d[i]; }
};

// ------------------------------------------------------------------
// General small dense matrix  (used only for SVD / LU inside solvePnP)
// ------------------------------------------------------------------
struct Mat {
    int rows=0,cols=0;
    std::vector<double> data;
    Mat(){}
    Mat(int r,int c):rows(r),cols(c),data(r*c,0.0){}
    double& at(int r,int c)       {return data[r*cols+c];}
    double  at(int r,int c) const {return data[r*cols+c];}
    static Mat eye(int n){Mat m(n,n);for(int i=0;i<n;i++)m.at(i,i)=1.0;return m;}
};

static Mat matMul(const Mat& A,const Mat& B){
    Mat C(A.rows,B.cols);
    for(int i=0;i<A.rows;i++) for(int k=0;k<A.cols;k++) for(int j=0;j<B.cols;j++)
        C.at(i,j)+=A.at(i,k)*B.at(k,j);
    return C;
}
static Mat matT(const Mat& A){
    Mat T(A.cols,A.rows);
    for(int i=0;i<A.rows;i++) for(int j=0;j<A.cols;j++) T.at(j,i)=A.at(i,j);
    return T;
}

// One-sided Jacobi SVD  (stable for <=20x20 matrices)
static void svdJacobi(const Mat& A,Mat& U,std::vector<double>& S,Mat& Vt){
    int m=A.rows,n=A.cols;
    Mat B=A; Mat V=Mat::eye(n);
    for(int sw=0;sw<100;sw++){
        double off=0;
        for(int p=0;p<n;p++) for(int q=p+1;q<n;q++){
            double bpp=0,bqq=0,bpq=0;
            for(int i=0;i<m;i++){bpp+=B.at(i,p)*B.at(i,p);bqq+=B.at(i,q)*B.at(i,q);bpq+=B.at(i,p)*B.at(i,q);}
            off+=bpq*bpq; if(std::fabs(bpq)<1e-15) continue;
            double tau=(bqq-bpp)/(2.0*bpq);
            double t=(tau>=0)?1.0/(tau+std::sqrt(1+tau*tau)):1.0/(tau-std::sqrt(1+tau*tau));
            double c=1.0/std::sqrt(1+t*t),s=c*t;
            for(int i=0;i<m;i++){double bp=B.at(i,p),bq=B.at(i,q);B.at(i,p)=c*bp-s*bq;B.at(i,q)=s*bp+c*bq;}
            for(int i=0;i<n;i++){double vp=V.at(i,p),vq=V.at(i,q);V.at(i,p)=c*vp-s*vq;V.at(i,q)=s*vp+c*vq;}
        }
        if(off<1e-28) break;
    }
    S.resize(n); U=Mat(m,n);
    for(int j=0;j<n;j++){
        double nr=0; for(int i=0;i<m;i++) nr+=B.at(i,j)*B.at(i,j); S[j]=std::sqrt(nr);
        if(S[j]>1e-15) for(int i=0;i<m;i++) U.at(i,j)=B.at(i,j)/S[j];
    }
    Vt=matT(V);
}

// Last right singular vector of A  (null-space approximation)
static Mat nullVec(const Mat& A){
    Mat U; std::vector<double> S; Mat Vt;
    svdJacobi(A,U,S,Vt);
    int idx=0; for(int i=1;i<(int)S.size();i++) if(S[i]<S[idx]) idx=i;
    Mat x(Vt.cols,1); for(int i=0;i<Vt.cols;i++) x.at(i,0)=Vt.at(idx,i);
    return x;
}

// LU solve with partial pivoting, in-place on b
static bool luSolve(Mat A,Mat& b){
    int n=A.rows;
    for(int col=0;col<n;col++){
        int best=col; for(int r=col+1;r<n;r++) if(std::fabs(A.at(r,col))>std::fabs(A.at(best,col))) best=r;
        if(std::fabs(A.at(best,col))<1e-14) return false;
        for(int j=0;j<n;j++) std::swap(A.at(col,j),A.at(best,j));
        for(int j=0;j<b.cols;j++) std::swap(b.at(col,j),b.at(best,j));
        for(int r=col+1;r<n;r++){
            double f=A.at(r,col)/A.at(col,col);
            for(int j=col;j<n;j++) A.at(r,j)-=f*A.at(col,j);
            for(int j=0;j<b.cols;j++) b.at(r,j)-=f*b.at(col,j);
        }
    }
    for(int r=n-1;r>=0;r--) for(int j=0;j<b.cols;j++){
        for(int c=r+1;c<n;c++) b.at(r,j)-=A.at(r,c)*b.at(c,j);
        b.at(r,j)/=A.at(r,r);
    }
    return true;
}

// ------------------------------------------------------------------
// Rodrigues: Vec3 rotation vector -> Mat33 rotation matrix
// ------------------------------------------------------------------
inline Mat33 rodrigues(const Vec3& r){
    double rx=r[0],ry=r[1],rz=r[2];
    double theta=std::sqrt(rx*rx+ry*ry+rz*rz);
    Mat33 R; if(theta<1e-10){R.d[0][0]=R.d[1][1]=R.d[2][2]=1.0;return R;}
    double c=std::cos(theta),s=std::sin(theta),c1=1-c;
    double nx=rx/theta,ny=ry/theta,nz=rz/theta;
    R.d[0][0]=c+nx*nx*c1;    R.d[0][1]=nx*ny*c1-nz*s; R.d[0][2]=nx*nz*c1+ny*s;
    R.d[1][0]=ny*nx*c1+nz*s; R.d[1][1]=c+ny*ny*c1;    R.d[1][2]=ny*nz*c1-nx*s;
    R.d[2][0]=nz*nx*c1-ny*s; R.d[2][1]=nz*ny*c1+nx*s; R.d[2][2]=c+nz*nz*c1;
    return R;
}

// Mat33 -> Vec3 (inverse Rodrigues)
inline Vec3 rodriguesInv(const Mat33& R){
    Vec3 r;
    double tr=R.d[0][0]+R.d[1][1]+R.d[2][2];
    double cosT=std::max(-1.0,std::min(1.0,(tr-1.0)/2.0));
    double theta=std::acos(cosT);
    if(std::fabs(theta)<1e-10) return r;
    double s=1.0/(2.0*std::sin(theta));
    r[0]=(R.d[2][1]-R.d[1][2])*s*theta;
    r[1]=(R.d[0][2]-R.d[2][0])*s*theta;
    r[2]=(R.d[1][0]-R.d[0][1])*s*theta;
    return r;
}

// Project nearest valid rotation matrix (SO3) via SVD
static Mat33 projectSO3(const Mat33& M){
    Mat A(3,3); for(int i=0;i<3;i++) for(int j=0;j<3;j++) A.at(i,j)=M.d[i][j];
    Mat U; std::vector<double> S; Mat Vt;
    svdJacobi(A,U,S,Vt);
    Mat Rm=matMul(U,Vt);
    double det=Rm.at(0,0)*(Rm.at(1,1)*Rm.at(2,2)-Rm.at(1,2)*Rm.at(2,1))
              -Rm.at(0,1)*(Rm.at(1,0)*Rm.at(2,2)-Rm.at(1,2)*Rm.at(2,0))
              +Rm.at(0,2)*(Rm.at(1,0)*Rm.at(2,1)-Rm.at(1,1)*Rm.at(2,0));
    if(det<0){for(int i=0;i<3;i++) U.at(i,2)*=-1; Rm=matMul(U,Vt);}
    Mat33 R; for(int i=0;i<3;i++) for(int j=0;j<3;j++) R.d[i][j]=Rm.at(i,j);
    return R;
}

// ------------------------------------------------------------------
// Internal helpers for the LM solver
// ------------------------------------------------------------------
static void computeResiduals(
    const double* p,
    const std::vector<Point3d>& obj, const std::vector<Point2d>& img,
    double fx,double fy,double cx,double cy,
    std::vector<double>& res)
{
    Vec3 rv={p[0],p[1],p[2]}, tv={p[3],p[4],p[5]};
    Mat33 R=rodrigues(rv);
    int n=(int)obj.size(); res.resize(2*n);
    for(int i=0;i<n;i++){
        double Xc=R.d[0][0]*obj[i].x+R.d[0][1]*obj[i].y+R.d[0][2]*obj[i].z+tv[0];
        double Yc=R.d[1][0]*obj[i].x+R.d[1][1]*obj[i].y+R.d[1][2]*obj[i].z+tv[1];
        double Zc=R.d[2][0]*obj[i].x+R.d[2][1]*obj[i].y+R.d[2][2]*obj[i].z+tv[2];
        if(std::fabs(Zc)<1e-10) Zc=1e-10;
        res[2*i  ]=fx*(Xc/Zc)+cx-img[i].x;
        res[2*i+1]=fy*(Yc/Zc)+cy-img[i].y;
    }
}

// Numerical Jacobian via central differences.
// This is the KEY fix: the previous analytic Jacobian had a sign error in
// the Rodrigues derivative that caused wild angle outputs. Numerical diff
// is unconditionally correct and costs only 12 extra residual evaluations.
static void computeJacobian(
    const double* p,
    const std::vector<Point3d>& obj, const std::vector<Point2d>& img,
    double fx,double fy,double cx,double cy,
    std::vector<double>& J, int n)
{
    const double eps=1e-5;
    J.resize(2*n*6);
    std::vector<double> rp,rm;
    double pp[6]; for(int k=0;k<6;k++) pp[k]=p[k];
    for(int col=0;col<6;col++){
        pp[col]=p[col]+eps; computeResiduals(pp,obj,img,fx,fy,cx,cy,rp);
        pp[col]=p[col]-eps; computeResiduals(pp,obj,img,fx,fy,cx,cy,rm);
        pp[col]=p[col];
        for(int row=0;row<2*n;row++) J[row*6+col]=(rp[row]-rm[row])/(2.0*eps);
    }
}

// ------------------------------------------------------------------
// solvePnP — public API
//   DLT initialisation then Levenberg-Marquardt refinement.
//   Returns true on success, false if initialisation fails.
// ------------------------------------------------------------------
inline bool solvePnP(
    const std::vector<Point3d>& objPts,
    const std::vector<Point2d>& imgPts,
    double fx, double fy, double cx, double cy,
    Vec3& rvec, Vec3& tvec)
{
    int n=(int)objPts.size();
    if(n<6) return false;

    // ---- DLT init ----
    Mat A(2*n,12);
    for(int i=0;i<n;i++){
        double u=(imgPts[i].x-cx)/fx, v=(imgPts[i].y-cy)/fy;
        double X=objPts[i].x,Y=objPts[i].y,Z=objPts[i].z;
        A.at(2*i,  0)=X;A.at(2*i,  1)=Y;A.at(2*i,  2)=Z;A.at(2*i,  3)=1;
        A.at(2*i,  8)=-u*X;A.at(2*i,  9)=-u*Y;A.at(2*i, 10)=-u*Z;A.at(2*i, 11)=-u;
        A.at(2*i+1,4)=X;A.at(2*i+1,5)=Y;A.at(2*i+1,6)=Z;A.at(2*i+1,7)=1;
        A.at(2*i+1,8)=-v*X;A.at(2*i+1,9)=-v*Y;A.at(2*i+1,10)=-v*Z;A.at(2*i+1,11)=-v;
    }
    Mat h=nullVec(A);
    double scale=0; for(int j=0;j<3;j++) scale+=h.at(j,0)*h.at(j,0);
    scale=std::sqrt(scale);
    if(scale<1e-12) return false;

    double sign=(h.at(11,0)/scale>0)?1.0:-1.0;
    Mat33 Rraw; Vec3 traw;
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++) Rraw.d[i][j]=sign*h.at(i*4+j,0)/scale;
        traw[i]=sign*h.at(i*4+3,0)/scale;
    }
    Mat33 Rso3=projectSO3(Rraw);
    rvec=rodriguesInv(Rso3); tvec=traw;

    // ---- Levenberg-Marquardt ----
    double params[6]={rvec[0],rvec[1],rvec[2],tvec[0],tvec[1],tvec[2]};
    double lambda=1e-3;

    std::vector<double> res,J;
    computeResiduals(params,objPts,imgPts,fx,fy,cx,cy,res);
    double cost=0; for(double r:res) cost+=r*r;

    for(int iter=0;iter<200;iter++){
        computeJacobian(params,objPts,imgPts,fx,fy,cx,cy,J,n);

        Mat JtJ(6,6); Mat Jtr(6,1);
        for(int row=0;row<2*n;row++){
            for(int a=0;a<6;a++){
                Jtr.at(a,0)-=J[row*6+a]*res[row];
                for(int b=0;b<6;b++) JtJ.at(a,b)+=J[row*6+a]*J[row*6+b];
            }
        }
        for(int k=0;k<6;k++) JtJ.at(k,k)*=(1.0+lambda);

        if(!luSolve(JtJ,Jtr)){lambda*=10.0;continue;}

        double newp[6]; for(int k=0;k<6;k++) newp[k]=params[k]+Jtr.at(k,0);
        std::vector<double> nr;
        computeResiduals(newp,objPts,imgPts,fx,fy,cx,cy,nr);
        double nc=0; for(double r:nr) nc+=r*r;

        if(nc<cost){
            for(int k=0;k<6;k++) params[k]=newp[k];
            res=nr; cost=nc;
            lambda=std::max(1e-10,lambda*0.1);
            double sn=0; for(int k=0;k<6;k++) sn+=Jtr.at(k,0)*Jtr.at(k,0);
            if(sn<1e-16) break;
        } else {
            lambda=std::min(1e10,lambda*10.0);
            if(lambda>1e9) break;
        }
    }

    rvec={params[0],params[1],params[2]};
    tvec={params[3],params[4],params[5]};
    return true;
}

// ------------------------------------------------------------------
// projectPoints — zero distortion assumed
// ------------------------------------------------------------------
inline std::vector<Point2d> projectPoints(
    const std::vector<Point3d>& pts,
    const Vec3& rvec,const Vec3& tvec,
    double fx,double fy,double cx,double cy)
{
    Mat33 R=rodrigues(rvec);
    std::vector<Point2d> out;
    for(auto& p:pts){
        double Xc=R.d[0][0]*p.x+R.d[0][1]*p.y+R.d[0][2]*p.z+tvec[0];
        double Yc=R.d[1][0]*p.x+R.d[1][1]*p.y+R.d[1][2]*p.z+tvec[1];
        double Zc=R.d[2][0]*p.x+R.d[2][1]*p.y+R.d[2][2]*p.z+tvec[2];
        if(std::fabs(Zc)<1e-10) Zc=1e-10;
        out.push_back({fx*(Xc/Zc)+cx,fy*(Yc/Zc)+cy});
    }
    return out;
}

// ------------------------------------------------------------------
// checkOrthonormal — RMSE of (R^T R - I); <1e-4 means valid rotation
// ------------------------------------------------------------------
inline double checkOrthonormal(const Mat33& R){
    double s=0;
    for(int i=0;i<3;i++) for(int j=0;j<3;j++){
        double v=0; for(int k=0;k<3;k++) v+=R.d[k][i]*R.d[k][j];
        double e=v-(i==j?1.0:0.0); s+=e*e;
    }
    return std::sqrt(s);
}

// ------------------------------------------------------------------
// eulerAngles — ZYX convention
//   pitch: nod up/down (X)   yaw: turn left/right (Y)   roll: tilt (Z)
//   All ~0 for a frontal face with the Y-corrected 3D model.
// ------------------------------------------------------------------
inline void eulerAngles(const Mat33& R,double& pitch,double& yaw,double& roll){
    pitch=std::atan2( R.d[2][1], R.d[2][2]);
    yaw  =std::atan2(-R.d[2][0], std::sqrt(R.d[2][1]*R.d[2][1]+R.d[2][2]*R.d[2][2]));
    roll =std::atan2( R.d[1][0], R.d[0][0]);
}

} // namespace pm