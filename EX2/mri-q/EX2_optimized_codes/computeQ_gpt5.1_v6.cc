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
                 float* __restrict phiR, float* __restrict phiI, float* __restrict phiMag) {
  int indexK;

#ifdef _OPENMP
#pragma omp parallel for default(none) private(indexK) shared(numK,phiR,phiI,phiMag)
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
            struct kValues * __restrict kVals,
            float* __restrict x, float* __restrict y, float* __restrict z,
            float * __restrict Qr, float * __restrict Qi) {

#ifdef _OPENMP
#pragma omp parallel default(none) shared(numK,numX,kVals,x,y,z,Qr,Qi)
  {
    int indexK, indexX;
#pragma omp for
    for (indexX = 0; indexX < numX; indexX++) {
      float x_val = x[indexX];
      float y_val = y[indexX];
      float z_val = z[indexX];
      float Qr_val = Qr[indexX];
      float Qi_val = Qi[indexX];

      for (indexK = 0; indexK < numK; indexK++) {
        float expArg = PIx2 * (kVals[indexK].Kx * x_val +
                               kVals[indexK].Ky * y_val +
                               kVals[indexK].Kz * z_val);

        float cosArg = cosf(expArg);
        float sinArg = sinf(expArg);

        float phi = kVals[indexK].PhiMag;
        Qr_val += phi * cosArg;
        Qi_val += phi * sinArg;
      }

      Qr[indexX] = Qr_val;
      Qi[indexX] = Qi_val;
    }
  }
#else
  int indexK, indexX;
  for (indexX = 0; indexX < numX; indexX++) {
    float x_val = x[indexX];
    float y_val = y[indexX];
    float z_val = z[indexX];
    float Qr_val = Qr[indexX];
    float Qi_val = Qi[indexX];

    for (indexK = 0; indexK < numK; indexK++) {
      float expArg = PIx2 * (kVals[indexK].Kx * x_val +
                             kVals[indexK].Ky * y_val +
                             kVals[indexK].Kz * z_val);

      float cosArg = cosf(expArg);
      float sinArg = sinf(expArg);

      float phi = kVals[indexK].PhiMag;
      Qr_val += phi * cosArg;
      Qi_val += phi * sinArg;
    }

    Qr[indexX] = Qr_val;
    Qi[indexX] = Qi_val;
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
