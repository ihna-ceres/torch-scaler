from setuptools import setup, find_packages

setup(
    name='torch-scaler',
    version='0.0.1',
    description='An alternative to scikit-learn scaler for pytorch tensor',
    author='ihna-ceres',
    author_email='ihna@cerestechs.com',
    url='https://github.com/ihna-ceres/torch-scaler',
    install_requires=['torch', 'numpy'],
    packages=find_packages(exclude=[]),
    keywords=['scaler', 'torch', 'pytorch', 'scikit-learn'],
    python_requires='>=3.6',
    package_data={},
    zip_safe=False
)
