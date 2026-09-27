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
  
  // Handle beta scaling first
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
  
  // Main computation with loop interchange and blocking
  const int BLOCK_SIZE = 64;
  
  for (int nn = 0; nn < n; ++nn) {
    const float *B_col = B + nn;
    float *C_col = C + nn * ldc;
    
    for (int kk = 0; kk < k; kk += BLOCK_SIZE) {
      int k_end = (kk + BLOCK_SIZE < k) ? (kk + BLOCK_SIZE) : k;
      
      for (int mm = 0; mm < m; mm += BLOCK_SIZE) {
        int m_end = (mm + BLOCK_SIZE < m) ? (mm + BLOCK_SIZE) : m;
        
        for (int i = kk; i < k_end; ++i) {
          const float *A_col = A + i * lda;
          float b_val = B_col[i * ldb];
          float alpha_b = alpha * b_val;
          
          for (int j = mm; j < m_end; ++j) {
            C_col[j] += alpha_b * A_col[j];
          }
        }
      }
    }
  }
}
