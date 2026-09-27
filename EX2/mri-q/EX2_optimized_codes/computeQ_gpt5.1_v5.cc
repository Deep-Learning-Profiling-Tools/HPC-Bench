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
                 float* phiR, float* phiI, float* phiMag) {
  int indexK = 0;
#ifdef _OPENMP
#pragma omp parallel for default(none) schedule(static) shared(phiR,phiI,phiMag,numK)
#endif
  for (indexK = 0; indexK < numK; indexK++) {
    const float real = phiR[indexK];
    const float imag = phiI[indexK];
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

  /* Parallelize outer K-loop; each thread owns private accumulators
     for all X and reduces once at the end for better locality. */
#ifdef _OPENMP
#pragma omp parallel default(none) \
  shared(numK,numX,kVals,x,y,z,Qr,Qi) \
  private(indexK,indexX)
  {
    /* Thread-local accumulation buffers to avoid false sharing and
       allow parallel writes without atomics. */
    float *localQr = (float *)malloc((size_t)numX * sizeof(float));
    float *localQi = (float *)malloc((size_t)numX * sizeof(float));
    if (localQr == NULL || localQi == NULL) {
      /* Fallback: if allocation fails, do work directly into global
         arrays, still respecting parallel correctness via critical. */
#pragma omp for schedule(static)
      for (indexK = 0; indexK < numK; indexK++) {
        const float kx = kVals[indexK].Kx;
        const float ky = kVals[indexK].Ky;
        const float kz = kVals[indexK].Kz;
        const float phi = kVals[indexK].PhiMag;
        for (indexX = 0; indexX < numX; indexX++) {
          const float expArg = PIx2 * (kx * x[indexX] +
                                       ky * y[indexX] +
                                       kz * z[indexX]);
          const float cosArg = cosf(expArg);
          const float sinArg = sinf(expArg);
#pragma omp atomic
          Qr[indexX] += phi * cosArg;
#pragma omp atomic
          Qi[indexX] += phi * sinArg;
        }
      }
      if (localQr) free(localQr);
      if (localQi) free(localQi);
    } else {
      /* Initialize local buffers */
      memset(localQr, 0, (size_t)numX * sizeof(float));
      memset(localQi, 0, (size_t)numX * sizeof(float));

#pragma omp for schedule(static)
      for (indexK = 0; indexK < numK; indexK++) {
        const float kx = kVals[indexK].Kx;
        const float ky = kVals[indexK].Ky;
        const float kz = kVals[indexK].Kz;
        const float phi = kVals[indexK].PhiMag;
        for (indexX = 0; indexX < numX; indexX++) {
          const float expArg = PIx2 * (kx * x[indexX] +
                                       ky * y[indexX] +
                                       kz * z[indexX]);
          const float cosArg = cosf(expArg);
          const float sinArg = sinf(expArg);
          localQr[indexX] += phi * cosArg;
          localQi[indexX] += phi * sinArg;
        }
      }

      /* Reduction into global Qr, Qi */
#pragma omp for schedule(static)
      for (indexX = 0; indexX < numX; indexX++) {
        Qr[indexX] += localQr[indexX];
        Qi[indexX] += localQi[indexX];
      }

      free(localQr);
      free(localQi);
    }
  }
#else
  /* Serial, cache-friendly implementation: keep kVals fields in locals
     and Qr/Qi streaming-friendly. */
  for (indexK = 0; indexK < numK; indexK++) {
    const float kx = kVals[indexK].Kx;
    const float ky = kVals[indexK].Ky;
    const float kz = kVals[indexK].Kz;
    const float phi = kVals[indexK].PhiMag;
    for (indexX = 0; indexX < numX; indexX++) {
      const float expArg = PIx2 * (kx * x[indexX] +
                                   ky * y[indexX] +
                                   kz * z[indexX]);
      const float cosArg = cosf(expArg);
      const float sinArg = sinf(expArg);
      Qr[indexX] += phi * cosArg;
      Qi[indexX] += phi * sinArg;
    }
  }
#endif
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
