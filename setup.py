import setuptools

setuptools.setup(
    name='ncp',
    version='1.0',
    include_package_data=True,
    packages=setuptools.find_packages(),
    install_requires=[
        'torch>=1.8',
        'thop',
    ]
)
