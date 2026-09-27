void basicSgemm( char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc )
{
  if ((transa != 'N') && (transa != 'n')) {
    std::cerr << "unsupported value of 'transa' in regtileSgemm()" << std::endl;
    return;
  }
  
  if ((transb != 'T') && (transb != 't')) {
    std::cerr << "unsupported value of 'transb' in regtileSgemm()" << std::endl;
    return;
  }
  
  // Pre-scale C matrix by beta
  if (beta == 0.0f) {
    for (int nn = 0; nn < n; ++nn) {
      for (int mm = 0; mm < m; ++mm) {
        C[mm + nn * ldc] = 0.0f;
      }
    }
  } else if (beta != 1.0f) {
    for (int nn = 0; nn < n; ++nn) {
      for (int mm = 0; mm < m; ++mm) {
        C[mm + nn * ldc] *= beta;
      }
    }
  }
  
  // Main computation with loop interchange and pointer arithmetic
  for (int nn = 0; nn < n; ++nn) {
    const float *B_col = B + nn;
    float *C_col = C + nn * ldc;
    
    for (int i = 0; i < k; ++i) {
      const float *A_col = A + i * lda;
      float b_val = alpha * B_col[i * ldb];
      
      for (int mm = 0; mm < m; ++mm) {
        C_col[mm] += A_col[mm] * b_val;
      }
    }
  }
}
