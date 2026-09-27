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
  int indexK;
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(numK, phiR, phiI, phiMag) private(indexK)
#endif
  for (indexK = 0; indexK < numK; indexK++) {
    float real = phiR[indexK];
    float imag = phiI[indexK];
    phiMag[indexK] = real * real + imag * imag;
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
#pragma omp parallel for default(none) private(indexK, indexX) shared(numK, numX, kVals, x, y, z, Qr, Qi)
#endif
  for (indexX = 0; indexX < numX; indexX++) {
    float x_val = x[indexX];
    float y_val = y[indexX];
    float z_val = z[indexX];
    float Qr_local = Qr[indexX];
    float Qi_local = Qi[indexX];

    for (indexK = 0; indexK < numK; indexK++) {
      const struct kValues kv = kVals[indexK];

      float expArg = PIx2 * (kv.Kx * x_val +
                             kv.Ky * y_val +
                             kv.Kz * z_val);

      float sinArg, cosArg;
      sincosf(expArg, &sinArg, &cosArg);

      float phi = kv.PhiMag;
      Qr_local += phi * cosArg;
      Qi_local += phi * sinArg;
    }

    Qr[indexX] = Qr_local;
    Qi[indexX] = Qi_local;
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
