#include <math.h>
#include <string.h>
#include <malloc.h>

#define PI   3.1415926535897932384626433832795029f
#define PIx2 6.2831853071795864769252867665590058f

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define K_ELEMS_PER_GRID 2048

struct kValues {
  float Kx;
  float Ky;
  float Kz;
  float PhiMag;
};

inline
void 
ComputePhiMagCPU(int numK, 
                 float* phiR, float* phiI, float* phiMag) {
  int indexK = 0;
  for (indexK = 0; indexK < numK; indexK++) {
    float real = phiR[indexK];
    float imag = phiI[indexK];
    phiMag[indexK] = real*real + imag*imag;
  }
}

inline
void
ComputeQCPU(int numK, int numX,
            struct kValues *kVals,
            float* x, float* y, float* z,
            float *Qr, float *Qi) {
  int indexK, indexX;
  
  for (indexX = 0; indexX < numX; indexX++) {
    float x_val = x[indexX];
    float y_val = y[indexX];
    float z_val = z[indexX];
    float Qr_acc = Qr[indexX];
    float Qr_acc2 = 0.0f;
    float Qi_acc = Qi[indexX];
    float Qi_acc2 = 0.0f;
    
    for (indexK = 0; indexK < numK - 1; indexK += 2) {
      float Kx0 = kVals[indexK].Kx;
      float Ky0 = kVals[indexK].Ky;
      float Kz0 = kVals[indexK].Kz;
      float phi0 = kVals[indexK].PhiMag;
      
      float Kx1 = kVals[indexK + 1].Kx;
      float Ky1 = kVals[indexK + 1].Ky;
      float Kz1 = kVals[indexK + 1].Kz;
      float phi1 = kVals[indexK + 1].PhiMag;
      
      float expArg0 = PIx2 * (Kx0 * x_val + Ky0 * y_val + Kz0 * z_val);
      float expArg1 = PIx2 * (Kx1 * x_val + Ky1 * y_val + Kz1 * z_val);
      
      float cosArg0 = cosf(expArg0);
      float sinArg0 = sinf(expArg0);
      float cosArg1 = cosf(expArg1);
      float sinArg1 = sinf(expArg1);
      
      Qr_acc += phi0 * cosArg0;
      Qi_acc += phi0 * sinArg0;
      Qr_acc2 += phi1 * cosArg1;
      Qi_acc2 += phi1 * sinArg1;
    }
    
    for (; indexK < numK; indexK++) {
      float Kx = kVals[indexK].Kx;
      float Ky = kVals[indexK].Ky;
      float Kz = kVals[indexK].Kz;
      float phi = kVals[indexK].PhiMag;
      
      float expArg = PIx2 * (Kx * x_val + Ky * y_val + Kz * z_val);
      float cosArg = cosf(expArg);
      float sinArg = sinf(expArg);
      
      Qr_acc += phi * cosArg;
      Qi_acc += phi * sinArg;
    }
    
    Qr[indexX] = Qr_acc + Qr_acc2;
    Qi[indexX] = Qi_acc + Qi_acc2;
  }
}

void createDataStructsCPU(int numK, int numX, float** phiMag,
	 float** Qr, float** Qi)
{
  *phiMag = (float* ) memalign(16, numK * sizeof(float));
  *Qr = (float*) memalign(16, numX * sizeof (float));
  memset((void *)*Qr, 0, numX * sizeof(float));
  *Qi = (float*) memalign(16, numX * sizeof (float));
  memset((void *)*Qi, 0, numX * sizeof(float));
}
