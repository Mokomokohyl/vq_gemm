from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='vq_gemm_cuda',
    ext_modules=[
        CUDAExtension(
            'vq_gemm_cuda',
            [
                'bind.cpp',
                'vq_gemm.cu',
            ],
            extra_cuda_cflags=[
                '-O3',
                '-arch=compute_120 -code=sm_120',
            ]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
    }
)