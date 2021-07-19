/mnt/home/miniconda3/bin/python setup.py sdist bdist_wheel
/mnt/home/miniconda3/bin/python -m twine upload --skip-existing dist/*
