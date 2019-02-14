from setuptools import setup, find_packages
setup(
  name = 'apprentice',
  version = '0.1.0',
  description = 'The apprentice',
  url = 'https://xgitlab.cels.anl.gov/mkrishnamoorthy/apprentice',
  author = 'Mohan Krishnamoorthy, Holger Schulz',
  author_email = 'mkrishnamoorthy@anl.gov, hschulz@fnal.gov',
  packages = find_packages(),
  include_package_data = True,
  install_requires = [
    'numpy',
    'scipy',
    'sklearn',
    'numba'
    # 'sobol',
    # 'pyDOE'
  ],
  scripts=[],
  extras_require = {
  },
  entry_points = {
  },
  dependency_links = [
  ]
)
