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

  /* Process 4 elements per iteration when possible for better ILP
     and fewer loop-control operations. */
  int limit = numK & ~3;
  for (; indexK < limit; indexK += 4) {
    float r0 = phiR[indexK];
    float i0 = phiI[indexK];
    float r1 = phiR[indexK + 1];
    float i1 = phiI[indexK + 1];
    float r2 = phiR[indexK + 2];
    float i2 = phiI[indexK + 2];
    float r3 = phiR[indexK + 3];
    float i3 = phiI[indexK + 3];

    phiMag[indexK]     = r0 * r0 + i0 * i0;
    phiMag[indexK + 1] = r1 * r1 + i1 * i1;
    phiMag[indexK + 2] = r2 * r2 + i2 * i2;
    phiMag[indexK + 3] = r3 * r3 + i3 * i3;
  }

  /* Handle remaining elements. */
  for (; indexK < numK; indexK++) {
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

  for (indexK = 0; indexK < numK; indexK++) {
    const float Kx   = kVals[indexK].Kx;
    const float Ky   = kVals[indexK].Ky;
    const float Kz   = kVals[indexK].Kz;
    const float phi  = kVals[indexK].PhiMag;

    float * __restrict Qr_loc = Qr;
    float * __restrict Qi_loc = Qi;
    float * __restrict x_loc  = x;
    float * __restrict y_loc  = y;
    float * __restrict z_loc  = z;

    /* Unroll the inner loop by 2 to reduce loop overhead and
       increase instruction-level parallelism. */
    int limit = numX & ~1;
    for (indexX = 0; indexX < limit; indexX += 2) {
      float x0 = x_loc[indexX];
      float y0 = y_loc[indexX];
      float z0 = z_loc[indexX];

      float expArg0 = PIx2 * (Kx * x0 + Ky * y0 + Kz * z0);
      float sinArg0, cosArg0;
      sincosf(expArg0, &sinArg0, &cosArg0);

      Qr_loc[indexX] += phi * cosArg0;
      Qi_loc[indexX] += phi * sinArg0;

      float x1 = x_loc[indexX + 1];
      float y1 = y_loc[indexX + 1];
      float z1 = z_loc[indexX + 1];

      float expArg1 = PIx2 * (Kx * x1 + Ky * y1 + Kz * z1);
      float sinArg1, cosArg1;
      sincosf(expArg1, &sinArg1, &cosArg1);

      Qr_loc[indexX + 1] += phi * cosArg1;
      Qi_loc[indexX + 1] += phi * sinArg1;
    }

    /* Handle remaining element if numX is odd. */
    for (; indexX < numX; indexX++) {
      float x0 = x_loc[indexX];
      float y0 = y_loc[indexX];
      float z0 = z_loc[indexX];

      float expArg = PIx2 * (Kx * x0 + Ky * y0 + Kz * z0);
      float sinArg, cosArg;
      sincosf(expArg, &sinArg, &cosArg);

      Qr_loc[indexX] += phi * cosArg;
      Qi_loc[indexX] += phi * sinArg;
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
