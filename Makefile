KERNELS ?= all


compile:
	KERNELS=$(KERNELS) python compile_kernels.py build_ext --inplace
run:
	KERNELS=$(KERNELS) python bench.py > ./logs/bench_$(KERNELS).log 2>&1
try:;$(MAKE) compile;$(MAKE) run

setup:
	mkdir -p logs
	$(MAKE) compile
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
try-s2-1024:;$(MAKE) try KERNELS=s2_1024
run-s2-1024:;$(MAKE) run KERNELS=s2_1024
try-s3:;$(MAKE) try KERNELS=s3
run-s3:;$(MAKE) run KERNELS=s3
run-gemm-128:; TEST_GEMM=TRUE KERNELS=s1 python bench.py > ./logs/bench_gemm_128.log 2>&1
run-gemm-1024:; TEST_GEMM=TRUE KERNELS=s2_1024 python bench.py > ./logs/bench_gemm_1024.log 2>&1

.PHONY: run compile try clean clean-logs try-s1 run-s1 try-s2-128 run-s2-128 try-s2-1024 run-s2-1024 try-s3 run-s3