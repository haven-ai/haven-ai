from setuptools import setup
import setuptools

setup(name='havenai-dev',
      version='0.6.0',
      description='Manage large-scale experiments',
      url='https://github.com/ElementAI/haven',
      maintainer='Issam Laradji',
      maintainer_email='issam.laradji@elementai.com',
      license='MIT',
      packages=setuptools.find_packages(),
      zip_safe=False,
      install_requires=[
        'tqdm>=0.0'
        'matplotlib>=0.0',
        'numpy>=0.0',
        'opencv-python-headless>=0.0',
        'pandas>=0.0',
        'Pillow>=0.0',
        'scikit-image>=0.0',
        'scikit-learn>=0.0',
        'scipy>=0.0',
        'sklearn>=0.0',
        'torch>=0.0',
        'torchvision>=0.0',
        'notebook >= 4.0'
      ]),