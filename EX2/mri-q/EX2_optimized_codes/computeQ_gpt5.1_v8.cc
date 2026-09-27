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

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (indexK = 0; indexK < numK; indexK++) {
    const float real = phiR[indexK];
    const float imag = phiI[indexK];
    phiMag[indexK] = real * real + imag * imag;
  }
}

inline
void
ComputeQCPU(int numK, int numX,
            struct kValues * __restrict kVals,
            float* __restrict x, float* __restrict y, float* __restrict z,
            float * __restrict Qr, float * __restrict Qi) {
  int indexK, indexX;

#ifdef _OPENMP
#pragma omp parallel for private(indexX) schedule(static)
#endif
  for (indexK = 0; indexK < numK; indexK++) {
    const float Kx = kVals[indexK].Kx;
    const float Ky = kVals[indexK].Ky;
    const float Kz = kVals[indexK].Kz;
    const float phi = kVals[indexK].PhiMag;

    for (indexX = 0; indexX < numX; indexX++) {
      const float expArg = PIx2 * (Kx * x[indexX] +
                                   Ky * y[indexX] +
                                   Kz * z[indexX]);

      const float cosArg = cosf(expArg);
      const float sinArg = sinf(expArg);

      Qr[indexX] += phi * cosArg;
      Qi[indexX] += phi * sinArg;
    }
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
