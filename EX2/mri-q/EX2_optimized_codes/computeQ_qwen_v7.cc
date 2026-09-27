#include <math.h>
#include <string.h>
#include <malloc.h>
#ifdef _OPENMP
#include <omp.h>
#endif

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
#ifdef _OPENMP
#pragma omp parallel for
#endif
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
#ifdef _OPENMP
#pragma omp parallel for private(indexX)
#endif
  for (indexK = 0; indexK < numK; indexK++) {
    for (indexX = 0; indexX < numX; indexX++) {
      float expArg = PIx2 * (kVals[indexK].Kx * x[indexX] +
                             kVals[indexK].Ky * y[indexX] +
                             kVals[indexK].Kz * z[indexX]);

      float cosArg;
      float sinArg;
      sincosf(expArg, &sinArg, &cosArg);

      float phi = kVals[indexK].PhiMag;
#ifdef _OPENMP
#pragma omp atomic
#endif
      Qr[indexX] += phi * cosArg;
#ifdef _OPENMP
#pragma omp atomic
#endif
      Qi[indexX] += phi * sinArg;
    }
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
