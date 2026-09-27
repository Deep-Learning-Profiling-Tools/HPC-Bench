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
  
  // Scale C by beta first if beta != 1.0
  if (beta != 1.0f) {
    for (int nn = 0; nn < n; ++nn) {
      for (int mm = 0; mm < m; ++mm) {
        C[mm + nn * ldc] *= beta;
      }
    }
  }
  
  // Compute alpha * A * B^T and accumulate into C
  for (int nn = 0; nn < n; ++nn) {
    const float *B_col = B + nn;
    for (int i = 0; i < k; ++i) {
      const float *A_col = A + i * lda;
      float b_val = B_col[i * ldb];
      float alpha_b = alpha * b_val;
      
      for (int mm = 0; mm < m; ++mm) {
        C[mm + nn * ldc] += alpha_b * A_col[mm];
      }
    }
  }
}
