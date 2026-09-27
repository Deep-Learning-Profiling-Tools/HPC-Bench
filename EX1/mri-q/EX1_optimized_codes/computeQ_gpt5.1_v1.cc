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
                 float* restrict phiR, float* restrict phiI, float* restrict phiMag) {
  int indexK;
  for (indexK = 0; indexK < numK; ++indexK) {
    float real = phiR[indexK];
    float imag = phiI[indexK];
    phiMag[indexK] = real * real + imag * imag;
  }
}

inline
void
ComputeQCPU(int numK, int numX,
            struct kValues * restrict kVals,
            float* restrict x, float* restrict y, float* restrict z,
            float * restrict Qr, float * restrict Qi) {

  int indexK, indexX;

  for (indexK = 0; indexK < numK; ++indexK) {
    const float kx = kVals[indexK].Kx;
    const float ky = kVals[indexK].Ky;
    const float kz = kVals[indexK].Kz;
    const float phi = kVals[indexK].PhiMag;

    for (indexX = 0; indexX < numX; ++indexX) {
      const float expArg =
        PIx2 * (kx * x[indexX] + ky * y[indexX] + kz * z[indexX]);

      float s, c;
      sincosf(expArg, &s, &c);

      Qr[indexX] += phi * c;
      Qi[indexX] += phi * s;
    }
  }
}

void createDataStructsCPU(int numK, int numX, float** phiMag,
	 float** Qr, float** Qi)
{
  (void)numK;
  *phiMag = (float* ) memalign(16, (size_t)numK * sizeof(float));
  *Qr = (float*) memalign(16, (size_t)numX * sizeof (float));
  memset((void *)*Qr, 0, (size_t)numX * sizeof(float));
  *Qi = (float*) memalign(16, (size_t)numX * sizeof (float));
  memset((void *)*Qi, 0, (size_t)numX * sizeof(float));
}
