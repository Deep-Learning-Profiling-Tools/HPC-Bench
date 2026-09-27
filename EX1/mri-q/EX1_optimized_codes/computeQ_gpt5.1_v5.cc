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

inline void ComputePhiMagCPU(int numK, float *phiR, float *phiI, float *phiMag) {
  int indexK;
  int limit = numK & ~3;

  for (indexK = 0; indexK < limit; indexK += 4) {
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

  for (; indexK < numK; ++indexK) {
    float real = phiR[indexK];
    float imag = phiI[indexK];
    phiMag[indexK] = real * real + imag * imag;
  }
}

inline void ComputeQCPU(int numK, int numX,
                        struct kValues *kVals,
                        float *x, float *y, float *z,
                        float *Qr, float *Qi) {

  int indexK, indexX;
  float scale = PIx2;

  for (indexX = 0; indexX < numX; ++indexX) {
    float xval = x[indexX];
    float yval = y[indexX];
    float zval = z[indexX];
    float qreal = Qr[indexX];
    float qimag = Qi[indexX];

    for (indexK = 0; indexK < numK; ++indexK) {
      const struct kValues kv = kVals[indexK];

      float expArg = scale * (kv.Kx * xval + kv.Ky * yval + kv.Kz * zval);

      float cosArg = cosf(expArg);
      float sinArg = sinf(expArg);

      float phi = kv.PhiMag;
      qreal += phi * cosArg;
      qimag += phi * sinArg;
    }

    Qr[indexX] = qreal;
    Qi[indexX] = qimag;
  }
}

void createDataStructsCPU(int numK, int numX, float **phiMag,
                          float **Qr, float **Qi) {
  (void)numK;

  *phiMag = (float *)memalign(16, numK * sizeof(float));

  *Qr = (float *)memalign(16, numX * sizeof(float));
  memset(*Qr, 0, (size_t)numX * sizeof(float));

  *Qi = (float *)memalign(16, numX * sizeof(float));
  memset(*Qi, 0, (size_t)numX * sizeof(float));
}
