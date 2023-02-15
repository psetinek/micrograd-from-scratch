try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='micrograd',
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      description="smth",
      long_description="smth",
      url='smth',
      author="Paul Setinek",
      author_email='paul.setinek@gmail.com',
      packages=['micrograd'])
