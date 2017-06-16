from setuptools import setup

setup(
    name='mltools',
    version='0.1.0',
    packages=['mltools'],
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
