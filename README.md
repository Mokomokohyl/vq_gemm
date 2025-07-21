## vq-gemm experiments

### Environment
conda env with torch and matplotlib

### Run tests
`<version>` can be `s1`, `s2-128`, `s2-1024`, `s3`
- `make compile KERNELS=<version>`: compile `<version>` kernel, generate vq_gemm_cuda_\<version\>.so   
`make compile` compile all kernels.
- `make try-<version>`: compile and run test of the corresponding version vq-gemm kernel.
- `make run-<version>`: run test of the corresponding version vq-gemm kernel without compile.
- `make run-gemm-128`: run benchmark of gemm in compiled `s1`
- `make run-gemm-1024`: run benchmark of gemm in compiled `s2_1024` (NOT IMPLMENTED YET)
- `make prof-s3`: run ncu profile of `s3`. ncu_reports will be in `ncu_reports/` directory.
    - Revise `run_s3.sh` to activate your Conda environment
    - Revise `NCU_LOG_NAME` in Makefile to determine name of the .ncu-rep output.

Terminal outputs are redirected to ./logs/bench_\<version\>.log  
Error heat map are generated in ./figures/

### TODO
- [ ] `s2-1024`
- [ ] optimize dequant kernel in `s1`
- [ ] optimize dequant func in `s3`.
- [ ] find how to use `setmaxnreg` correctly