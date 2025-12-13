from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='sparse_mask',
    version='0.1.0',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='sparse_mask_lib',
            sources=['sparse_mask/csrc/sparse_mask.cu'],
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
    },
    python_requires='>=3.8',
    install_requires=[
        'torch',
    ],
)
