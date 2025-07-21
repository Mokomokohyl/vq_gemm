import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Compile flags set. rm build/ before revise it and compile
base_nvcc_flags = ['-O2', '-arch=sm_120a', '-lineinfo']
# base_nvcc_flags = ['-O2', '-arch=sm_120a', '-lineinfo', '--maxrregcount=128']

kernels_to_compile_str = os.getenv('KERNELS', 'all')
nvcc_flags = {
    'cublas_gemm': base_nvcc_flags + ['-arch=sm_120a'],
    's1': base_nvcc_flags,
    's2_128': base_nvcc_flags,
    's2_512': base_nvcc_flags,
    's3_naive': base_nvcc_flags + ['--maxrregcount=128'],
    's3_wasp': base_nvcc_flags + ['--maxrregcount=128'],
}
source = {
    'cublas_gemm': [
        'bind.cpp',
        './ref_gemm/cublas_gemm_fp16.cu',
    ],
    's1':   [
                'bind.cpp',
                './strategy_1/src.cu',
            ], 
    's2_128':
            [
                'bind.cpp',
                './strategy_2/bsz128/src.cu',
            ],
    's2_512':
            [
                'bind.cpp',
                './strategy_2/bsz512/src.cu',
            ],
    's3_naive':
            [
                'bind.cpp',
                './strategy_3/naive/src.cu',
            ],
    's3_wasp':
            [
                'bind.cpp',
                './strategy_3/wasp/src.cu',
            ]
}


if kernels_to_compile_str == 'all':
    kernels_to_compile = list(source.keys())
else:
    kernels_to_compile = [k.strip() for k in kernels_to_compile_str.split(',')]

for kernels_to_compile_str in kernels_to_compile:
    setup(
        name=f'vq_gemm_cuda_{kernels_to_compile_str}',
        ext_modules=[
            CUDAExtension(
                name=f'vq_gemm_cuda_{kernels_to_compile_str}',
                sources=source[kernels_to_compile_str],
                extra_compile_args={'cxx': ['-O2'], 'nvcc': nvcc_flags[kernels_to_compile_str]}
            )
        ],
        cmdclass={
            'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
        }
    )