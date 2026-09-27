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
  
  for (int nn = 0; nn < n; ++nn) {
    for (int mm = 0; mm < m; ++mm) {
      float c = 0.0f;
      const float *a_ptr = &A[mm];
      const float *b_ptr = &B[nn];
      
      for (int i = 0; i < k; ++i) {
        c += (*a_ptr) * (*b_ptr);
        a_ptr += lda;
        b_ptr += ldb;
      }
      
      float *c_ptr = &C[mm + nn*ldc];
      *c_ptr = (*c_ptr) * beta + alpha * c;
    }
  }
}
