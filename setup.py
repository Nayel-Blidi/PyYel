from setuptools import setup, find_packages

setup(
    name='PyYel',
    version='0.0.9',
    author='Nayel Maxime BLIDI',
    author_email='nayel.blidi@ipsa.fr',
    description='A data science library that allows easy and quick machine learning solutions deployement.',
    long_description="A data science library that allows easy and quick machine learning solutions deployement.",
    long_description_content_type='text/markdown',
    url='https://github.com/Nayel-Blidi/PyYel',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        # List your dependencies here
        'numpy',
        'pandas',
        'scikit-learn',
        'configparser',
        'tk',
        'opencv-python'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.6',
)
