## vq-gemm experiments

### Environment
conda env with torch and matplotlib

### Run tests

Before getting started, run `make setup` to compile reference GeMM (cublas pure fp16).  
Then run `make compile` to compile all kernels.

`<version>` can be `s1`, `s2-128`, `s2-512`, `s3-naive`, `s3-wasp`
- `make compile KERNELS=<version>`: compile `<version>` kernel, generate vq_gemm_cuda_\<version\>.so   
`make compile` compile all kernels.
- `make try-<version>`: compile and run test of the corresponding version vq-gemm kernel.
- `make run-<version>`: run test of the corresponding version vq-gemm kernel without compile.
- `make run-gemm-128`: run benchmark of gemm in compiled `s1`
- `make run-gemm-512`: run benchmark of gemm in compiled `s2-512`
- `make prof-<version>`: run ncu profile of \<version\> vq-gemm kernel. ncu_reports will be in `ncu_reports/` directory.
    - Revise prof_cmd to activate your conda env    

Terminal outputs are redirected to ./logs/bench_\<version\>.log  
Error heat map are generated in ./figures/

## TODO
- [ ] optimize `s2-128`. Currently for M, N, K = (4096, 2048, 4096), `s1` has 195.45TFLOPS while `s2` has only 156.292TFLOPS