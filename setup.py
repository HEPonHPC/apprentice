from setuptools import setup, find_packages
setup(
  name = 'apprentice',
  version = '0.5.1',
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
   'numba',
   'h5py'
 ],
  scripts=["bin/app-ls", "bin/app-tune", "bin/app-build", "bin/app-rox", "bin/app-nest", "bin/app-rox-parallel", "bin/app-predict", "bin/app-sample", "etc/convertData.py", "etc/simplecomp.py", "etc/convert.py"],
  extras_require = {
  },
  entry_points = {
  },
  dependency_links = [
  ]
)
