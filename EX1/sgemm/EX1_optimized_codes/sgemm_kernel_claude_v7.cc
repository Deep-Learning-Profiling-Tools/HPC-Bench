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
  
  // Handle beta scaling separately for better cache behavior
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
  
  // Main computation with loop reordering and blocking
  const int BLOCK_SIZE = 32;
  
  for (int nn = 0; nn < n; ++nn) {
    for (int mm_block = 0; mm_block < m; mm_block += BLOCK_SIZE) {
      int mm_end = (mm_block + BLOCK_SIZE < m) ? mm_block + BLOCK_SIZE : m;
      
      for (int i = 0; i < k; ++i) {
        float b_val = B[nn + i * ldb];
        float alpha_b = alpha * b_val;
        const float *A_col = A + i * lda;
        
        for (int mm = mm_block; mm < mm_end; ++mm) {
          C[mm + nn * ldc] += alpha_b * A_col[mm];
        }
      }
    }
  }
}
