#!/bin/bash
source /home/ylhuang/miniconda3/bin/activate
conda activate fiber
cd /home/ylhuang/vq_gemm
PROFILING=TRUE KERNELS=s3 python bench.py