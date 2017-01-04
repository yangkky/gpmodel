from setuptools import setup

setup(name='gpmodel',
      version='1.0',
      url='http://github.com/yangkky/gpmodel',
      description='Classes for Gaussian process modeling of proteins.',
      packages=['gpmodel'],
      license='MIT',
      author='Kevin Yang',
      author_email='seinchin@gmail.com',
      test_suite='nose.collector',
      tests_require=['nose'])
