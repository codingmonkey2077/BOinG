#I only tested it under python 3.7, am best install with anaconda environement

#conda install gxx_linux-64 gcc_linux-64 swig
pip install -r requirements.txt

#(very important here, as here I rewrite some functions in pyrfr)
cd Dependency
cd rfr 
bash build_packages.sh 
