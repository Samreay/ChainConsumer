pip uninstall -y chainconsumer
cd .. && python setup.py install && cd doc && make clean && make rst && make htmlfull