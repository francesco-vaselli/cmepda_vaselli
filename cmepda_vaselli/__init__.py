import os

PACKAGE_NAME = 'cmepda_vaselli'

"""Basic folder structure of the package.
"""
ROOT = os.path.abspath(os.path.dirname(__file__))
BASE = os.path.join(ROOT, os.pardir)
BASIC_CLASS = os.path.join(ROOT, 'basic_classification')
IMAGE_CLASS = os.path.join(ROOT, 'image_classification')
NPR_CLASS = os.path.join(ROOT, 'NPRnet_classification')
UTILS = os.path.join(ROOT, 'utils')
