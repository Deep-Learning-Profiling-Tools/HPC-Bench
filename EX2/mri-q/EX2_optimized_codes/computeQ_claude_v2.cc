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
  #pragma omp parallel for simd schedule(static)
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
  int indexX;

#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (indexX = 0; indexX < numX; indexX++) {
    float Qr_local = 0.0f;
    float Qi_local = 0.0f;
    float x_val = x[indexX];
    float y_val = y[indexX];
    float z_val = z[indexX];

    int indexK;
    for (indexK = 0; indexK < numK; indexK++) {
      float Kx = kVals[indexK].Kx;
      float Ky = kVals[indexK].Ky;
      float Kz = kVals[indexK].Kz;
      float phi = kVals[indexK].PhiMag;

      float expArg = PIx2 * (Kx * x_val + Ky * y_val + Kz * z_val);

      float cosArg = cosf(expArg);
      float sinArg = sinf(expArg);

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
