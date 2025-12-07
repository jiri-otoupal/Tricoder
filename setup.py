from setuptools import setup, find_packages

setup(
    name='tvci',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'scikit-learn>=1.0.0',
        'gensim>=4.0.0',
        'annoy>=1.17.0',
    ],
    python_requires='>=3.7',
)

