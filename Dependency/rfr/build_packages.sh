set -ex

cd build
rm -rf *
cmake ..
make pyrfr_docstrings
cd python_package
pip install . --user
