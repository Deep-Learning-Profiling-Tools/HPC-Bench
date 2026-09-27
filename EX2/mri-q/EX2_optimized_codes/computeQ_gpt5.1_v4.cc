#include <math.h>
#include <string.h>
#include <malloc.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define PI   3.1415926535897932384626433832795029f
#define PIx2 6.2831853071795864769252867665590058f

#define MIN(X,Y) ((X) < (Y) ? (X) : (MIN(Y,X)))
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
                 float* __restrict phiR, float* __restrict phiI, float* __restrict phiMag) {
  int indexK;
  #pragma omp parallel for if(numK > 1024) schedule(static)
  for (indexK = 0; indexK < numK; indexK++) {
    float real = phiR[indexK];
    float imag = phiI[indexK];
    phiMag[indexK] = real*real + imag*imag;
  }
}

inline
void
ComputeQCPU(int numK, int numX,
            struct kValues * __restrict kVals,
            float* __restrict x, float* __restrict y, float* __restrict z,
            float * __restrict Qr, float * __restrict Qi) {
  int indexK, indexX;

  #pragma omp parallel for if(numK > 1 && numX > 1) schedule(static) private(indexX)
  for (indexX = 0; indexX < numX; indexX++) {
    float x_val = x[indexX];
    float y_val = y[indexX];
    float z_val = z[indexX];

    float sumQr = 0.0f;
    float sumQi = 0.0f;

    for (indexK = 0; indexK < numK; indexK++) {
      float kx = kVals[indexK].Kx;
      float ky = kVals[indexK].Ky;
      float kz = kVals[indexK].Kz;
      float phi = kVals[indexK].PhiMag;

      float expArg = PIx2 * (kx * x_val + ky * y_val + kz * z_val);

      float sinArg, cosArg;
      sincosf(expArg, &sinArg, &cosArg);

      sumQr += phi * cosArg;
      sumQi += phi * sinArg;
    }

    Qr[indexX] += sumQr;
    Qi[indexX] += sumQi;
  }
}

void createDataStructsCPU(int numK, int numX, float** phiMag,
	 float** Qr, float** Qi)
{
  *phiMag = (float* ) memalign(16, (size_t)numK * sizeof(float));
  *Qr = (float*) memalign(16, (size_t)numX * sizeof (float));
  memset((void *)*Qr, 0, (size_t)numX * sizeof(float));
  *Qi = (float*) memalign(16, (size_t)numX * sizeof (float));
  memset((void *)*Qi, 0, (size_t)numX * sizeof(float));
}
