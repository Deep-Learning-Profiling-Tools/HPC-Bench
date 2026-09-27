void basicSgemm(char transa, char transb,
                int m, int n, int k,
                float alpha, const float *A, int lda,
                const float *B, int ldb,
                float beta, float *C, int ldc)
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
    float *c_col       = C + nn * ldc;
    const float *b_col = B + nn * ldb;

    for (int mm = 0; mm < m; ++mm) {
      float acc = 0.0f;

      const float *a_row = A + mm;
      const float *b_ptr = b_col;

      for (int i = 0; i < k; ++i) {
        acc += (*a_row) * (*b_ptr);
        a_row += lda;
        b_ptr += 1;
      }

      c_col[mm] = c_col[mm] * beta + alpha * acc;
    }
  }
}
