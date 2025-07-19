import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

kernels_to_compile_str = os.getenv('KERNELS', 'all')
source = {
    's1':   [
                'bind.cpp',
                './strategy_1/src.cu',
            ], 
    's2_128':
            [
                'bind.cpp',
                './strategy_2/bsz128/src.cu',
            ],
    's2_1024':
            [
                'bind.cpp',
                './strategy_2/bsz1024/src.cu',
            ],
    's3':
            [
                'bind.cpp',
                './strategy_3/src.cu',
            ]
}

base_nvcc_flags = ['-O2', '-arch=sm_120a', '--maxrregcount=128']

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
                extra_compile_args={'cxx': ['-O2'], 'nvcc': base_nvcc_flags}
            )
        ],
        cmdclass={
            'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
        }
    )