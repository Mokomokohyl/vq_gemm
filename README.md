## vq-gemm experiments

### Environment
conda env with torch and matplotlib

### Run tests
`<version>` can be `s1`, `s2-128`, `s2-512`, `s3-naive`, `s3-wasp`
- `make compile KERNELS=<version>`: compile `<version>` kernel, generate vq_gemm_cuda_\<version\>.so   
`make compile` compile all kernels.
- `make try-<version>`: compile and run test of the corresponding version vq-gemm kernel.
- `make run-<version>`: run test of the corresponding version vq-gemm kernel without compile.
- `make run-gemm-128`: run benchmark of gemm in compiled `s1`
- `make run-gemm-1024`: run benchmark of gemm in compiled `s2_1024` (NOT IMPLMENTED YET)
- `make prof-<version>`: run ncu profile of \<version\> vq-gemm kernel. ncu_reports will be in `ncu_reports/` directory.
    - Revise prof_cmd to activate your conda env    

Terminal outputs are redirected to ./logs/bench_\<version\>.log  
Error heat map are generated in ./figures/

## TODO
- [ ] Fix dequant kernel of `s1`. Theoretically, current `s1` only accepts N = 128x, but it output wrong when N == 768
- [ ] optimize `s2-128`. Currently for M, N, K = (4096, 2048, 4096), `s1` has 195.45TFLOPS while `s2` has only 156.292TFLOPS
- [ ] `s2-512`
- [ ] `s3-wasp`