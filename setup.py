from setuptools import setup, find_packages


__version__ = '0.0.1'

setup(
    name='cross_view_transformer',
    version=__version__,
    author='Brady Zhou',
    author_email='brady.zhou@utexas.edu',
    url='https://github.com/bradyz/cross_view_transformers',
    license='MIT',
    packages=find_packages(include=['cross_view_transformer', 'cross_view_transformer.*']),
    zip_safe=False,
)
