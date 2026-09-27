# Benchmark Catalog

HPC-Bench contains 92 CPU workloads, evaluated in the serial (EX1) and OpenMP (EX2) settings, and 22 Rodinia-derived CUDA workloads, evaluated in the CUDA (EX3) setting. The CPU workloads span 16 computational motifs.

## Source suites

| Suite | CPU workloads | Reference |
|:---|---:|:---|
| PolyBench | 30 | L.-N. Pouchet. *PolyBench/C: The Polyhedral Benchmark Suite*. https://www.cs.colostate.edu/~pouchet/software/polybench/ |
| Rodinia | 19 | S. Che et al. *Rodinia: A Benchmark Suite for Heterogeneous Computing*. IISWC 2009. https://www.cs.virginia.edu/rodinia/ |
| MachSuite | 18 | B. Reagen et al. *MachSuite: Benchmarks for Accelerator Design and Customized Architectures*. IISWC 2014. https://github.com/breagen/MachSuite |
| Parboil | 10 | J. A. Stratton et al. *Parboil: A Revised Benchmark Suite for Scientific and Commercial Throughput Computing*. IMPACT-12-01, 2012. http://impact.crhc.illinois.edu/parboil/parboil.aspx |
| MiBench | 9 | M. R. Guthaus et al. *MiBench: A Free, Commercially Representative Embedded Benchmark Suite*. WWC 2001. https://vhosts.eecs.umich.edu/mibench/ |
| Codee | 5 | Codee open catalog of benchmarks. https://www.codee.com/ |
| PARSEC | 1 | C. Bienia et al. *The PARSEC Benchmark Suite: Characterization and Architectural Implications*. PACT 2008. https://parsec.cs.princeton.edu/ |

## Computational motifs

| Abbrev. | Motif | Workloads |
|:---|:---|---:|
| DLA | Dense Linear Algebra | 22 |
| SC | Stencil / Scientific Computing | 13 |
| IVP | Image and Video Processing | 7 |
| GA | Graph Algorithms | 6 |
| NBM | N-body Methods | 6 |
| DP | Dynamic Programming | 5 |
| SS | Sorting / Search | 5 |
| CML | Clustering / Machine Learning | 4 |
| MSC | Medical / Scientific Computing | 4 |
| STC | Statistical Computations | 4 |
| UB | Utility Benchmarks | 4 |
| MCM | Monte Carlo Methods | 3 |
| SLA | Sparse Linear Algebra | 3 |
| CE | Cryptography / Encoding | 2 |
| CFD | Computational Fluid Dynamics | 2 |
| SM | Spectral Methods | 2 |

Motif labels are defined in `scripts/analysis/motif.py`.

## Difficulty levels

Each workload is assigned a structural difficulty level from its annotated kernel region, using branch count, lines of code, maximum loop depth, and function count (`scripts/analysis/difficulty_scores.py`). The suite has 61 workloads at d1, 17 at d2, 9 at d3, and 5 at d4. The level measures structural complexity of the code to be optimized, not the available optimization headroom.

## CPU workloads (EX1 and EX2)

| Benchmark | Suite | Motif | Difficulty |
|:---|:---|:---|:---|
| `2mm` | PolyBench | DLA | d1 |
| `3mm` | PolyBench | DLA | d1 |
| `adi` | PolyBench | SC | d1 |
| `aes` | MachSuite | CE | d1 |
| `atax` | PolyBench | DLA | d1 |
| `atmux` | Codee | UB | d1 |
| `b+tree` | Rodinia | GA | d3 |
| `backprop` | Rodinia | CML | d1 |
| `basicmath-mibench` | MiBench | UB | d1 |
| `bfs` | Rodinia | GA | d3 |
| `bfs-bulk-mach` | MachSuite | GA | d1 |
| `bfs-queue-mach` | MachSuite | GA | d2 |
| `bicg` | PolyBench | DLA | d1 |
| `bilateral` | Rodinia | IVP | d1 |
| `bitcount-mibench` | MiBench | UB | d1 |
| `canneal-parsec` | PARSEC | CML | d1 |
| `cfd` | Rodinia | CFD | d2 |
| `cholesky` | PolyBench | DLA | d1 |
| `correlation` | PolyBench | STC | d1 |
| `coulomb` | Codee | NBM | d1 |
| `covariance` | PolyBench | STC | d1 |
| `cutcp` | Parboil | NBM | d4 |
| `deriche` | PolyBench | IVP | d1 |
| `dijkstra-mibench` | MiBench | GA | d1 |
| `doitgen` | PolyBench | STC | d1 |
| `durbin` | PolyBench | STC | d1 |
| `fdtd-2d` | PolyBench | SC | d1 |
| `fft-strided-mach` | MachSuite | SM | d1 |
| `fft-transpose-mach` | MachSuite | SM | d2 |
| `floyd-warshall` | PolyBench | DP | d1 |
| `gemm` | PolyBench | DLA | d1 |
| `gemm-blocked-mach` | MachSuite | DLA | d1 |
| `gemm-ncubed-mach` | MachSuite | DLA | d1 |
| `gemver` | PolyBench | DLA | d1 |
| `gesummv` | PolyBench | DLA | d1 |
| `gramschmidt` | PolyBench | DLA | d1 |
| `haccmk` | Codee | NBM | d1 |
| `heartwall` | Rodinia | MSC | d4 |
| `heat-3d` | PolyBench | SC | d1 |
| `histo` | Parboil | UB | d2 |
| `hotspot` | Rodinia | SC | d3 |
| `hotspot3D` | Rodinia | SC | d2 |
| `jacobi-1d` | PolyBench | SC | d1 |
| `jacobi-2d` | PolyBench | SC | d1 |
| `kmeans` | Rodinia | CML | d2 |
| `kmp-mach` | MachSuite | SS | d2 |
| `lavaMD` | Rodinia | NBM | d1 |
| `lbm` | Parboil | CFD | d4 |
| `leukocyte` | Rodinia | IVP | d1 |
| `lu` | PolyBench | DLA | d1 |
| `lud` | Rodinia | DLA | d1 |
| `ludcmp` | PolyBench | DLA | d1 |
| `matmul` | Codee | DLA | d1 |
| `md-grid-mach` | MachSuite | NBM | d2 |
| `md-knn-mach` | MachSuite | NBM | d1 |
| `mri-gridding` | Parboil | MSC | d3 |
| `mri-q` | Parboil | MSC | d1 |
| `mvt` | PolyBench | DLA | d1 |
| `myocyte` | Rodinia | MSC | d1 |
| `nussinov` | PolyBench | DP | d2 |
| `nw` | Rodinia | DP | d3 |
| `nw-mach` | MachSuite | DP | d2 |
| `particlefilter` | Rodinia | MCM | d2 |
| `pathfinder` | Rodinia | DP | d2 |
| `patricia-mibench` | MiBench | GA | d2 |
| `pi` | Codee | MCM | d2 |
| `qsort-mibench` | MiBench | SS | d3 |
| `sad` | Parboil | IVP | d3 |
| `seidel-2d` | PolyBench | SC | d1 |
| `sgemm` | Parboil | DLA | d1 |
| `sort-merge-mach` | MachSuite | SS | d1 |
| `sort-radix-mach` | MachSuite | SS | d1 |
| `spmv` | Parboil | SLA | d1 |
| `spmv-crs-mach` | MachSuite | SLA | d1 |
| `spmv-ellpack-mach` | MachSuite | SLA | d1 |
| `srad_v1` | Rodinia | SC | d2 |
| `srad_v2` | Rodinia | SC | d3 |
| `stencil` | Parboil | SC | d1 |
| `stencil-2d-mach` | MachSuite | SC | d1 |
| `stencil-3d-mach` | MachSuite | SC | d1 |
| `streamcluster` | Rodinia | CML | d3 |
| `stringsearch-mibench` | MiBench | SS | d1 |
| `susan-c-mibench` | MiBench | IVP | d4 |
| `susan-e-mibench` | MiBench | IVP | d4 |
| `susan-s-mibench` | MiBench | IVP | d2 |
| `symm` | PolyBench | DLA | d1 |
| `syr2k` | PolyBench | DLA | d1 |
| `syrk` | PolyBench | DLA | d1 |
| `tpacf` | Parboil | MCM | d2 |
| `trisolv` | PolyBench | DLA | d1 |
| `trmm` | PolyBench | DLA | d1 |
| `viterbi-mach` | MachSuite | CE | d1 |

## CUDA workloads (EX3)

`b+tree`, `bfs`, `bilateral`, `cfd`, `dwt2d`, `gaussian`, `heartwall`, `hotspot`, `hotspot3D`, `huffman`, `hybridsort`, `kmeans`, `lavaMD`, `leukocyte`, `lud`, `myocyte`, `nn`, `particlefilter`, `pathfinder`, `srad_v1`, `srad_v2`, `streamcluster`.

`dwt2d`, `gaussian`, `huffman`, `hybridsort`, and `nn` are GPU-oriented Rodinia workloads that are not part of the CPU set.

## Licensing

The benchmark programs and input data retain the licenses of their upstream suites. The MIT license of this repository does not cover them; see the upstream `COPYING` or `README` files in `EX{1,2,3}/<benchmark>/`.
