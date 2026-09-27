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
  
  // Precompute alpha * beta multiplication to reduce computation in the loop
  const float alpha_beta = alpha * beta;
  
  // Loop interchange and blocking optimizations
  const int block_size = 32; // Tunable parameter
  
  for (int mm_block = 0; mm_block < m; mm_block += block_size) {
    for (int nn_block = 0; nn_block < n; nn_block += block_size) {
      for (int mm = mm_block; mm < mm_block + block_size && mm < m; ++mm) {
        for (int nn = nn_block; nn < nn_block + block_size && nn < n; ++nn) {
          float c = 0.0f;
          const float *a_ptr = &A[mm];
          const float *b_ptr = &B[nn];
          
          // Unroll inner loop by 4
          int i = 0;
          for (; i <= k - 4; i += 4) {
            c += a_ptr[i * lda] * b_ptr[i * ldb];
            c += a_ptr[(i+1) * lda] * b_ptr[(i+1) * ldb];
            c += a_ptr[(i+2) * lda] * b_ptr[(i+2) * ldb];
            c += a_ptr[(i+3) * lda] * b_ptr[(i+3) * ldb];
          }
          
          // Handle remaining iterations
          for (; i < k; ++i) {
            c += a_ptr[i * lda] * b_ptr[i * ldb];
          }
          
          // Use precomputed value
          C[mm + nn * ldc] = C[mm + nn * ldc] * beta + alpha * c;
        }
      }
    }
  }
}
