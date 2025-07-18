KERNELS ?= all
compile:
	KERNELS=$(KERNELS) python compile_kernels.py build_ext --inplace
run:
	python bench.py > bench.log 2>&1
try:
	python compile_kernels.py build_ext --inplace
	python bench.py > bench.log 2>&1
clean:
	@rm -r build
	@rm ./*.so

.PHONY: run compile try clean