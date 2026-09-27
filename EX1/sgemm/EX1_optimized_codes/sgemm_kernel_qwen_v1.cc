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
  
  // Register blocking optimization
  const int TILE_SIZE = 32;
  for (int mm = 0; mm < m; mm += TILE_SIZE) {
    for (int nn = 0; nn < n; nn += TILE_SIZE) {
      for (int i = 0; i < k; ++i) {
        // Process tiles of C
        for (int mm2 = mm; mm2 < mm + TILE_SIZE && mm2 < m; ++mm2) {
          float a = A[mm2 + i * lda];
          for (int nn2 = nn; nn2 < nn + TILE_SIZE && nn2 < n; ++nn2) {
            float b = B[nn2 + i * ldb];
            C[mm2 + nn2 * ldc] = C[mm2 + nn2 * ldc] * beta + alpha * a * b;
          }
        }
      }
    }
  }
}
