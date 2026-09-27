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
  
  for (int mm = 0; mm < m; ++mm) {
    for (int nn = 0; nn < n; ++nn) {
      float c = 0.0f;
      const float *A_row = A + mm;
      const float *B_col = B + nn;
      for (int i = 0; i < k; ++i) {
        c += A_row[i * lda] * B_col[i * ldb];
      }
      C[mm+nn*ldc] = C[mm+nn*ldc] * beta + alpha * c;
    }
  }
}
