from setuptools import setup, Extension

setup(
    name='fastops',
    ext_modules=[
        Extension(
            'fastops',
            sources=['fastops.c'],
            extra_compile_args=['-O3', '-march=native'],
        ),
    ],
)
