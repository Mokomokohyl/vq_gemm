KERNELS ?= all


compile:
	KERNELS=$(KERNELS) python compile_kernels.py build_ext --inplace
run:
	KERNELS=$(KERNELS) python bench.py > ./logs/bench_$(KERNELS).log 2>&1
try:
	KERNELS=$(KERNELS) python compile_kernels.py build_ext --inplace
	KERNELS=$(KERNELS) python bench.py > ./logs/bench_$(KERNELS).log 2>&1

setup:
	mkdir -p logs
	$(MAKE) compile
clean:
	@rm -r build
	@rm ./*.so

clean-logs:
	@rm ./figures/*.png
	@rm ./logs/*.log

try-s1:
	$(MAKE) try KERNELS=s1
run-s1:
	$(MAKE) run KERNELS=s1
try-s2-128:
	$(MAKE) try KERNELS=s2_128
run-s2-128:
	$(MAKE) run KERNELS=s2_128
try-s3:
	$(MAKE) try KERNELS=s3
run-s3:
	$(MAKE) run KERNELS=s3

.PHONY: run compile try clean try-s1