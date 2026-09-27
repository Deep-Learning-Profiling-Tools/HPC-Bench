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
  int indexK;
  const float * __restrict phiR_p = phiR;
  const float * __restrict phiI_p = phiI;
  float * __restrict phiMag_p = phiMag;

  for (indexK = 0; indexK < numK; indexK++) {
    const float real = phiR_p[indexK];
    const float imag = phiI_p[indexK];
    phiMag_p[indexK] = real*real + imag*imag;
  }
}

inline
void
ComputeQCPU(int numK, int numX,
            struct kValues *kVals,
            float* x, float* y, float* z,
            float *Qr, float *Qi) {
  int indexK, indexX;

  const struct kValues * __restrict kVals_p = kVals;
  const float * __restrict x_p = x;
  const float * __restrict y_p = y;
  const float * __restrict z_p = z;
  float * __restrict Qr_p = Qr;
  float * __restrict Qi_p = Qi;

  for (indexK = 0; indexK < numK; indexK++) {
    const float Kx = kVals_p[indexK].Kx;
    const float Ky = kVals_p[indexK].Ky;
    const float Kz = kVals_p[indexK].Kz;
    const float phi = kVals_p[indexK].PhiMag;

    for (indexX = 0; indexX < numX; indexX++) {
      const float xx = x_p[indexX];
      const float yy = y_p[indexX];
      const float zz = z_p[indexX];

      const float expArg = PIx2 * (Kx * xx + Ky * yy + Kz * zz);

      const float cosArg = cosf(expArg);
      const float sinArg = sinf(expArg);

      Qr_p[indexX] += phi * cosArg;
      Qi_p[indexX] += phi * sinArg;
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
