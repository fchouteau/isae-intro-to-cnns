"""
PEP 517 doesn’t support editable installs
so this file is currently here to support "pip install -e ."
"""
import setuptools

if __name__ == '__main__':
    setuptools.setup()
