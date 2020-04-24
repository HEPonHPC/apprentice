from setuptools import setup, find_packages
setup(
  name = 'apprentice',
  version = '1.0.2',
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
   'h5py',
   'mpi4py'
 ],
  scripts=["bin/app-ls", "bin/app-tune2", "bin/app-build", "bin/app-predict", "bin/app-yoda2h5", "bin/app-sample", "etc/convertData.py", "etc/convert.py", "etc/extrema.py"],
  extras_require = {
  },
  entry_points = {
  },
  dependency_links = [
  ]
)
