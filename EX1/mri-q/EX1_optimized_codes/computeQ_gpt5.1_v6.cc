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
  for (; indexK + 3 < numK; indexK += 4) {
    float real0 = phiR[indexK];
    float imag0 = phiI[indexK];
    float real1 = phiR[indexK + 1];
    float imag1 = phiI[indexK + 1];
    float real2 = phiR[indexK + 2];
    float imag2 = phiI[indexK + 2];
    float real3 = phiR[indexK + 3];
    float imag3 = phiI[indexK + 3];

    phiMag[indexK]     = real0 * real0 + imag0 * imag0;
    phiMag[indexK + 1] = real1 * real1 + imag1 * imag1;
    phiMag[indexK + 2] = real2 * real2 + imag2 * imag2;
    phiMag[indexK + 3] = real3 * real3 + imag3 * imag3;
  }

  for (; indexK < numK; indexK++) {
    float real = phiR[indexK];
    float imag = phiI[indexK];
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
  for (indexK = 0; indexK < numK; indexK++) {
    const float Kx = kVals[indexK].Kx;
    const float Ky = kVals[indexK].Ky;
    const float Kz = kVals[indexK].Kz;
    const float phi = kVals[indexK].PhiMag;

    for (indexX = 0; indexX + 3 < numX; indexX += 4) {
      const float x0 = x[indexX];
      const float y0 = y[indexX];
      const float z0 = z[indexX];

      const float x1 = x[indexX + 1];
      const float y1 = y[indexX + 1];
      const float z1 = z[indexX + 1];

      const float x2 = x[indexX + 2];
      const float y2 = y[indexX + 2];
      const float z2 = z[indexX + 2];

      const float x3 = x[indexX + 3];
      const float y3 = y[indexX + 3];
      const float z3 = z[indexX + 3];

      const float expArg0 = PIx2 * (Kx * x0 + Ky * y0 + Kz * z0);
      const float expArg1 = PIx2 * (Kx * x1 + Ky * y1 + Kz * z1);
      const float expArg2 = PIx2 * (Kx * x2 + Ky * y2 + Kz * z2);
      const float expArg3 = PIx2 * (Kx * x3 + Ky * y3 + Kz * z3);

      const float cosArg0 = cosf(expArg0);
      const float sinArg0 = sinf(expArg0);
      const float cosArg1 = cosf(expArg1);
      const float sinArg1 = sinf(expArg1);
      const float cosArg2 = cosf(expArg2);
      const float sinArg2 = sinf(expArg2);
      const float cosArg3 = cosf(expArg3);
      const float sinArg3 = sinf(expArg3);

      Qr[indexX]     += phi * cosArg0;
      Qi[indexX]     += phi * sinArg0;
      Qr[indexX + 1] += phi * cosArg1;
      Qi[indexX + 1] += phi * sinArg1;
      Qr[indexX + 2] += phi * cosArg2;
      Qi[indexX + 2] += phi * sinArg2;
      Qr[indexX + 3] += phi * cosArg3;
      Qi[indexX + 3] += phi * sinArg3;
    }

    for (; indexX < numX; indexX++) {
      float expArg = PIx2 * (Kx * x[indexX] +
                             Ky * y[indexX] +
                             Kz * z[indexX]);

      float cosArg = cosf(expArg);
      float sinArg = sinf(expArg);

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
