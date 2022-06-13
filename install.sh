#!/bin/bash
systemName=$(uname -a)

echo =======================================================================================
echo Install Solver for this Poject
echo =======================================================================================

echo Starting installation process...
if [[ $systemName == *"Darwin"* ]]; then
  export MACOSX_DEPLOYMENT_TARGET=12.0
fi
echo ===============================================================================
#echo Step 1: Install numba and pybind11
echo Step 1: Install packages used in the repository
pip3 install -r requirements.txt
#pip3 install plotly

echo ===============================================================================
echo Step 2: Install model solution core

echo "Do you want to install eigen3 solver?"
echo -n "y/n: "
read -r EIGEN

if [[ $EIGEN = "y" ]]
then
	pip3 install ./src/cppcore
fi
echo ===============================================================================
echo Step 3: Install C kernels for petsc4py
pip3 install ./src/linearsystemcore

