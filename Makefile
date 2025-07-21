SHELL := /bin/bash
KERNELS ?= all
NCU_LOG_NAME = $(KERNELS)

NCU = /usr/local/cuda-12.8/bin/ncu

# source activate
CONDA_ACTIVATE = /home/ylhuang/miniconda3/bin/activate

CONDA_ENV_NAME = fiber

compile:
	mkdir -p logs
	KERNELS=$(KERNELS) python compile_kernels.py build_ext --inplace
run:
	KERNELS=$(KERNELS) python bench.py > ./logs/bench_$(KERNELS).log 2>&1
try:;$(MAKE) compile;$(MAKE) run

profile: # revise to activate your conda env
	mkdir -p ncu_reports
	rm -f ./ncu_reports/$(NCU_LOG_NAME).ncu-rep
	PROFILING=TRUE KERNELS=$(KERNELS) source $(CONDA_ACTIVATE) && conda activate $(CONDA_ENV_NAME) && cd /home/ylhuang/vq_gemm && \
	$(NCU) --import-source yes --set full -o ./ncu_reports/$(NCU_LOG_NAME) python bench.py > ./logs/ncu_$(KERNELS)_ouptput.log

clean:
	@rm -r build
	@rm ./*.so

clean-logs:
	@rm ./figures/*.png
	@rm ./logs/*.log

try-s1:;$(MAKE) try KERNELS=s1
run-s1:;$(MAKE) run KERNELS=s1

try-s2-128:;$(MAKE) try KERNELS=s2_128
run-s2-128:;$(MAKE) run KERNELS=s2_128

try-s2-512:;$(MAKE) try KERNELS=s2_512
run-s2-512:;$(MAKE) run KERNELS=s2_512

try-s3-naive:;$(MAKE) try KERNELS=s3_naive
run-s3-naive:;$(MAKE) run KERNELS=s3_naive

try-s3-wasp:;$(MAKE) try KERNELS=s3_wasp
run-s3-wasp:;$(MAKE) run KERNELS=s3_wasp

prof-s1:;$(MAKE) profile KERNELS=s1
prof-s2-128:;$(MAKE) profile KERNELS=s2_128
prof-s2-512:;$(MAKE) profile KERNELS=s2_512
prof-s3-naive:;$(MAKE) profile KERNELS=s3_naive
prof-s3-wasp:;$(MAKE) profile KERNELS=s3_wasp

run-gemm-128:; TEST_GEMM=TRUE KERNELS=s1 python bench.py > ./logs/bench_gemm_128.log 2>&1
run-gemm-s2:; TEST_GEMM=TRUE KERNELS=s2_128 python bench.py > ./logs/bench_gemm_128.log 2>&1
run-gemm-512:; TEST_GEMM=TRUE KERNELS=s2_512 python bench.py > ./logs/bench_gemm_512.log 2>&1

.PHONY: run compile try clean clean-logs try-s1 run-s1 try-s2-128 run-s2-128 try-s2-1024 run-s2-1024 try-s3 run-s3