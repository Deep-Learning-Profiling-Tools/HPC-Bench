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
    if (beta == 0.0f) {
      for (int nn = 0; nn < n; ++nn) {
        for (int mm = 0; mm < m; ++mm) {
          C[mm + nn * ldc] = 0.0f;
        }
      }
    } else {
      for (int nn = 0; nn < n; ++nn) {
        for (int mm = 0; mm < m; ++mm) {
          C[mm + nn * ldc] *= beta;
        }
      }
    }
  }
  
  // Compute alpha * A * B^T and add to C
  for (int nn = 0; nn < n; ++nn) {
    const float *B_col = B + nn;
    for (int mm = 0; mm < m; ++mm) {
      const float *A_row = A + mm;
      float c = 0.0f;
      
      // Unroll the inner loop by 4
      int i = 0;
      for (; i + 3 < k; i += 4) {
        float a0 = A_row[i * lda];
        float a1 = A_row[(i + 1) * lda];
        float a2 = A_row[(i + 2) * lda];
        float a3 = A_row[(i + 3) * lda];
        
        float b0 = B_col[i * ldb];
        float b1 = B_col[(i + 1) * ldb];
        float b2 = B_col[(i + 2) * ldb];
        float b3 = B_col[(i + 3) * ldb];
        
        c += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
      }
      
      // Handle remaining elements
      for (; i < k; ++i) {
        c += A_row[i * lda] * B_col[i * ldb];
      }
      
      C[mm + nn * ldc] += alpha * c;
    }
  }
}
