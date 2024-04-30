from setuptools import setup, find_packages
try:
    from install_preserve import preserve
except ImportError:
    import pip  # noqa
    pip.main(['install', 'install-preserve'])
    from install_preserve import preserve  # noqa

install_requires = [
    'imagehash',
    'efficientnet_pytorch',
    'numpy',
    'Pillow',
    'torch>=2.0.0',
    'torchvision>=0.17.0',
]

exclusions = [
    'torch',
    'torchvision',
]

install_requires = preserve(install_requires, exclusions, verbose=True)


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name='antidupe',
    version='0.0.1',
    packages=find_packages(),
    install_requires=install_requires,
    entry_points={
        'console_scripts': [
        ],
    },
    author='Manbehindthemadness',
    author_email='manbehindthemadness@gmail.com',
    description='Image deduplicator using CNN, Cosine Similarity, Image Hashing, Structural Similarity Index '
                'Measurement, and Euclidean Distance',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/manbehindthemadness/antidupe',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)