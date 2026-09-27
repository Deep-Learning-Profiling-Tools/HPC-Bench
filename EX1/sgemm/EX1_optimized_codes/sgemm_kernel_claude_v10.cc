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
  
  // Precompute alpha and beta conditions
  const int use_alpha = (alpha != 1.0f);
  const int use_beta = (beta != 0.0f);
  const int beta_is_one = (beta == 1.0f);
  
  for (int nn = 0; nn < n; ++nn) {
    const float *B_col = B + nn;
    float *C_col = C + nn * ldc;
    
    for (int mm = 0; mm < m; ++mm) {
      const float *A_row = A + mm;
      const float *B_ptr = B_col;
      
      float c = 0.0f;
      
      // Loop unrolling with 4-way accumulation
      int i = 0;
      for (; i + 3 < k; i += 4) {
        float a0 = A_row[i * lda];
        float a1 = A_row[(i + 1) * lda];
        float a2 = A_row[(i + 2) * lda];
        float a3 = A_row[(i + 3) * lda];
        
        float b0 = B_ptr[i * ldb];
        float b1 = B_ptr[(i + 1) * ldb];
        float b2 = B_ptr[(i + 2) * ldb];
        float b3 = B_ptr[(i + 3) * ldb];
        
        c += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
      }
      
      // Handle remaining elements
      for (; i < k; ++i) {
        c += A_row[i * lda] * B_ptr[i * ldb];
      }
      
      // Compute final result with conditional branches reduced
      float result;
      if (use_beta) {
        if (beta_is_one) {
          result = C_col[mm] + (use_alpha ? alpha * c : c);
        } else {
          result = C_col[mm] * beta + (use_alpha ? alpha * c : c);
        }
      } else {
        result = use_alpha ? alpha * c : c;
      }
      
      C_col[mm] = result;
    }
  }
}
