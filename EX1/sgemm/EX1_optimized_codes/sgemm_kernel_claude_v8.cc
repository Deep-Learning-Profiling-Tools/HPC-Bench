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
  
  // Handle beta scaling first if beta != 1.0
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
  
  // Blocking parameters for cache optimization
  const int BM = 64;
  const int BN = 64;
  const int BK = 64;
  
  for (int nn_block = 0; nn_block < n; nn_block += BN) {
    int nn_end = (nn_block + BN < n) ? nn_block + BN : n;
    
    for (int mm_block = 0; mm_block < m; mm_block += BM) {
      int mm_end = (mm_block + BM < m) ? mm_block + BM : m;
      
      for (int kk_block = 0; kk_block < k; kk_block += BK) {
        int kk_end = (kk_block + BK < k) ? kk_block + BK : k;
        
        for (int nn = nn_block; nn < nn_end; ++nn) {
          const float *B_col = B + nn;
          float *C_col = C + nn * ldc;
          
          for (int mm = mm_block; mm < mm_end; ++mm) {
            float c = 0.0f;
            const float *A_row = A + mm;
            
            // Unroll inner loop by 4
            int i = kk_block;
            for (; i + 3 < kk_end; i += 4) {
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
            for (; i < kk_end; ++i) {
              float a = A_row[i * lda];
              float b = B_col[i * ldb];
              c += a * b;
            }
            
            C_col[mm] += alpha * c;
          }
        }
      }
    }
  }
}
