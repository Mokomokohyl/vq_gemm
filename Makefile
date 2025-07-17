compile:
	python compile_kernels.py build_ext --inplace
run:
	python bench.py > bench.log 2>&1
clean:
	@rm -r build
	@rm ./*.so

.PHONY: run compile 