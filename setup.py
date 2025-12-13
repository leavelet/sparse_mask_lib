from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='sparse_mask_lib',
    ext_modules=[
        CUDAExtension(
            name='sparse_mask_lib',
            sources=['csrc/sparse_mask.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-gencode', 'arch=compute_90,code=sm_90a', # H100
                    '-gencode', 'arch=compute_100,code=sm_100a', # B200
                    '-gencode', 'arch=compute_103,code=sm_103a', # B300
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)