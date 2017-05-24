from setuptools import setup

setup(
    name='ml-tools',
    version='0.1.0',
    packages=['mltoolz'],
    url='https://gyorgy.orosz.link',
    license='MIT',
    install_requires=[
        'scikit-learn',
        # 'matplotlib',
        'pandas',
        'spacy',
        'scipy',
        'numpy',
    ],
    author='Gyorgy Orosz',
    author_email='gyorgy@orosz.link',
    description='Common utility functions for daily data science work with sklearn, spacy, pandas, etc.'
)
