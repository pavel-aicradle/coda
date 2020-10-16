from setuptools import setup, find_packages
import sys
sys.path.append('coda')
import coda

with open('README.md') as readme:
	long_description = readme.read()

setup(name='TODO', # Not sure what this should be just yet
	version=coda.__version__,
	description='Implementation of the Community Outlier Detection Algorithm',
	long_description=long_description,
	long_description_content_type='text/markdown',
	url='https://github.com/TODO/coda', # sort this out
	author='Pavel Komarov',
	license='BSD',
	packages=[x for x in find_packages() if 'tests' not in x],
	install_requires=['pytest', 'numpy', 'pandas'], # any others?
	author_email='pvlkmrv@gmail.com')