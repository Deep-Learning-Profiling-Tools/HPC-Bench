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
                 float* __restrict phiR, 
                 float* __restrict phiI, 
                 float* __restrict phiMag) {
  int indexK = 0;
  for (indexK = 0; indexK < numK; indexK++) {
    float real = phiR[indexK];
    float imag = phiI[indexK];
    phiMag[indexK] = real * real + imag * imag;
  }
}

inline
void
ComputeQCPU(int numK, int numX,
            struct kValues * __restrict kVals,
            float* __restrict x, 
            float* __restrict y, 
            float* __restrict z,
            float * __restrict Qr, 
            float * __restrict Qi) {
  int indexK, indexX;

  for (indexK = 0; indexK < numK; indexK++) {
    const float Kx     = kVals[indexK].Kx;
    const float Ky     = kVals[indexK].Ky;
    const float Kz     = kVals[indexK].Kz;
    const float phi    = kVals[indexK].PhiMag;
    const float twoPiKx = PIx2 * Kx;
    const float twoPiKy = PIx2 * Ky;
    const float twoPiKz = PIx2 * Kz;

    for (indexX = 0; indexX < numX; indexX++) {
      const float expArg = twoPiKx * x[indexX] +
                           twoPiKy * y[indexX] +
                           twoPiKz * z[indexX];

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
  (void)numK; /* numK is only used to size phiMag; kept for interface compatibility */

  *phiMag = (float* ) memalign(16, (size_t)numK * sizeof(float));
  *Qr = (float*) memalign(16, (size_t)numX * sizeof (float));
  memset((void *)*Qr, 0, (size_t)numX * sizeof(float));
  *Qi = (float*) memalign(16, (size_t)numX * sizeof (float));
  memset((void *)*Qi, 0, (size_t)numX * sizeof(float));
}
