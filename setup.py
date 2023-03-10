from distutils.core import setup
setup(name='pointProcess', 
    version='1.0', 
    author='Guillaume Becq', 
    author_email='guillaume.becq@gipsa-lab.grenoble-inp.fr', 
    description='Point Process Simulation', 
    long_description='Classes and functions for point processes simulation, statistics computation and observation. ',
    licenses='CeCILL',
    keywords=['point processes', 'Poisson processes'], 
    data_files=['LICENSE.txt'], 
    py_modules=['pointProcess'], 
    url='http://www.gipsa-lab.grenoble-inp.fr/~guillaume.becq/projets.html',
    platforms=['Windows', 'Mac', 'Linux'], 
    depends=['numpy', 'scipy', 'matplotlib'], 
    )
