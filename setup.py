from setuptools import setup, find_packages
setup(
  name = 'apprentice',
  version = '0.0.1',
  description = 'The apprentice',
  url = 'https://xgitlab.cels.anl.gov/mkrishnamoorthy/apprentice',
  author = 'Mohan Krishnamoorthy',
  author_email = 'mkrishnamoorthy@anl.gov',
  packages = find_packages(),
  include_package_data = True,
  install_requires = [
    'numpy',
    'scipy',
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
