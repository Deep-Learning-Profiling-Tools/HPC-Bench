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
  
  // Scale C by beta first
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
  
  // Perform matrix multiplication with loop reordering and blocking
  const int block_size = 32;
  
  for (int nn = 0; nn < n; nn += block_size) {
    int n_block = (nn + block_size > n) ? (n - nn) : block_size;
    
    for (int mm = 0; mm < m; mm += block_size) {
      int m_block = (mm + block_size > m) ? (m - mm) : block_size;
      
      for (int kk = 0; kk < k; kk += block_size) {
        int k_block = (kk + block_size > k) ? (k - kk) : block_size;
        
        for (int n_idx = nn; n_idx < nn + n_block; ++n_idx) {
          const float *B_col = B + n_idx * ldb + kk;
          
          for (int m_idx = mm; m_idx < mm + m_block; ++m_idx) {
            const float *A_row = A + m_idx + kk * lda;
            float sum = 0.0f;
            
            for (int k_idx = 0; k_idx < k_block; ++k_idx) {
              sum += A_row[k_idx * lda] * B_col[k_idx];
            }
            
            C[m_idx + n_idx * ldc] += alpha * sum;
          }
        }
      }
    }
  }
}
