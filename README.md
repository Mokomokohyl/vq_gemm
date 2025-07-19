## vq-gemm experiments

### Environment
conda env with torch and matplotlib

### Run tests
- `make compile KERNELS=<version>`: compile `<version>` kernel. `make compile` compile all kernels.
- `make try-<version>`: compile and run test of corresponding version vq-gemm kernel.
- `make run-<version>`: run test of corresponding version vq-gemm kernel without compile.
- `<version>` can be `s1`, `s2-128`, `s2-1024`, `s3`

### TODO
- [ ] `s2-1024`
- [ ] optimize dequant kernel in `s1`
- [ ] optimize dequant func in `s3`.