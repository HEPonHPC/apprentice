from setuptools import setup, find_packages
setup(
  name = 'pyapprentice',
  version = '1.1.0',
  description = 'The apprentice',
  url = 'https://computing.fnal.gov/hep-on-hpc/',
  author = 'Mohan Krishnamoorthy, Holger Schulz',
  author_email = 'mkrishnamoorthy@anl.gov, iamholger@gmail.com',
   packages = find_packages(),
  include_package_data = True,
 install_requires = [
   'numpy>=1.15.0',
   'scipy>=1.7.2',
   'pyomo>=6.4.0',
   # 'h5py>=2.8.0',
   'matplotlib>=3.0.0',
   # 'GPy>=1.9.9'
   # 'mpi4py>=3.0.0',
 ],
  scripts=["bin/app-ls", "bin/app-tune2", "bin/app-build", "bin/app-predict", "bin/app-datadirtojson", "bin/app-yoda2h5", "bin/app-yodaenvelope", "bin/app-sample", "etc/extrema.py"],
  extras_require = {
  },
  entry_points = {
  },
  dependency_links = [
  ]
)
