from setuptools import setup, find_packages

setup(
    name='torch-scaler',
    version='0.0.2',
    description='A scaler similar to the scikit-learn scaler that works on pytorch tensors',
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
