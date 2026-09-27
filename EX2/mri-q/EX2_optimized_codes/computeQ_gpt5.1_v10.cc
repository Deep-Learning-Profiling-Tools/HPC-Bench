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
  /* Parallelize over spatial samples (indexX) to avoid races on Qr/Qi */
#pragma omp parallel default(none) shared(numK, numX, kVals, x, y, z, Qr, Qi) private(indexK, indexX)
  {
#pragma omp for
#endif
  for (indexX = 0; indexX < numX; indexX++) {
    const float x_val = x[indexX];
    const float y_val = y[indexX];
    const float z_val = z[indexX];
    float q_r = Qr[indexX];
    float q_i = Qi[indexX];

    for (indexK = 0; indexK < numK; indexK++) {
      const struct kValues kv = kVals[indexK];
      const float expArg =
        PIx2 * (kv.Kx * x_val +
                kv.Ky * y_val +
                kv.Kz * z_val);

      const float cosArg = cosf(expArg);
      const float sinArg = sinf(expArg);

      const float phi = kv.PhiMag;
      q_r += phi * cosArg;
      q_i += phi * sinArg;
    }

    Qr[indexX] = q_r;
    Qi[indexX] = q_i;
  }
#ifdef _OPENMP
  }
#endif
}

void createDataStructsCPU(int numK, int numX, float** phiMag,
	 float** Qr, float** Qi)
{
  (void)numK; /* unused here, kept for interface completeness */
  *phiMag = (float* ) memalign(16, (size_t)numK * sizeof(float));
  *Qr = (float*) memalign(16, (size_t)numX * sizeof (float));
  memset((void *)*Qr, 0, (size_t)numX * sizeof(float));
  *Qi = (float*) memalign(16, (size_t)numX * sizeof (float));
  memset((void *)*Qi, 0, (size_t)numX * sizeof(float));
}
