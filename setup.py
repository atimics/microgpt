from setuptools import setup, Extension
import sys
import platform

# Platform-specific compiler flags
extra_compile_args = []
extra_link_args = []

if sys.platform == 'win32':
    # MSVC compiler flags
    extra_compile_args = ['/O2', '/fp:fast']
else:
    # GCC/Clang flags (Linux, macOS)
    extra_compile_args = ['-O3', '-march=native', '-ffast-math']
    
    # Link flags
    if sys.platform == 'linux':
        # Linux has libmvec (vector math library)
        extra_link_args = ['-lmvec', '-lm']
    else:
        # macOS doesn't need explicit -lm, doesn't have -lmvec
        extra_link_args = []

setup(
    name='fastops',
    ext_modules=[
        Extension(
            'fastops',
            sources=['fastops.c'],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        ),
    ],
)
