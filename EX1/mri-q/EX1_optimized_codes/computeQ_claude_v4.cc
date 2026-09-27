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
    float Qr_acc = Qr[indexX];
    float Qi_acc = Qi[indexX];
    float x_val = x[indexX];
    float y_val = y[indexX];
    float z_val = z[indexX];
    
    for (indexK = 0; indexK < numK; indexK++) {
      float expArg = PIx2 * (kVals[indexK].Kx * x_val +
                             kVals[indexK].Ky * y_val +
                             kVals[indexK].Kz * z_val);

      float cosArg = cosf(expArg);
      float sinArg = sinf(expArg);

      float phi = kVals[indexK].PhiMag;
      Qr_acc += phi * cosArg;
      Qi_acc += phi * sinArg;
    }
    
    Qr[indexX] = Qr_acc;
    Qi[indexX] = Qi_acc;
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
