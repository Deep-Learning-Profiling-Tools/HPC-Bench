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
                 float* __restrict phiR, float* __restrict phiI, float* __restrict phiMag) {
  int indexK = 0;
  for (; indexK + 3 < numK; indexK += 4) {
    const float real0 = phiR[indexK];
    const float imag0 = phiI[indexK];
    const float real1 = phiR[indexK + 1];
    const float imag1 = phiI[indexK + 1];
    const float real2 = phiR[indexK + 2];
    const float imag2 = phiI[indexK + 2];
    const float real3 = phiR[indexK + 3];
    const float imag3 = phiI[indexK + 3];

    phiMag[indexK]     = real0 * real0 + imag0 * imag0;
    phiMag[indexK + 1] = real1 * real1 + imag1 * imag1;
    phiMag[indexK + 2] = real2 * real2 + imag2 * imag2;
    phiMag[indexK + 3] = real3 * real3 + imag3 * imag3;
  }

  for (; indexK < numK; ++indexK) {
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
  for (indexK = 0; indexK < numK; ++indexK) {
    const float Kx     = kVals[indexK].Kx;
    const float Ky     = kVals[indexK].Ky;
    const float Kz     = kVals[indexK].Kz;
    const float phi    = kVals[indexK].PhiMag;

    for (indexX = 0; indexX < numX; ++indexX) {
      const float x_val = x[indexX];
      const float y_val = y[indexX];
      const float z_val = z[indexX];

      const float expArg = PIx2 * (Kx * x_val +
                                   Ky * y_val +
                                   Kz * z_val);

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
  *Qr     = (float*)  memalign(16, (size_t)numX * sizeof(float));
  memset((void *)*Qr, 0, (size_t)numX * sizeof(float));
  *Qi     = (float*)  memalign(16, (size_t)numX * sizeof(float));
  memset((void *)*Qi, 0, (size_t)numX * sizeof(float));
}
