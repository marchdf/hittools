try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='ho_apriori',
      version='0.1',
      description='A-priori study of high order numerical methods for LES',
      long_description=readme(),
      keywords='high order numerical methods, a-priori, turbulence, computational fluid dynamics',
      url='https://github.com/marchdf/ho_apriori',
      download_url='https://github.com/marchdf/ho_apriori',
      author='Marc T. Henry de Frahan',
      author_email='marchdf@gmail.com',
      license='Apache License 2.0',
      packages=['ho_apriori'],
      install_requires=[
          'numpy',
          'pandas',
          'scipy',
          'sphinx_rtd_theme'
      ],
      test_suite='tests',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False)
