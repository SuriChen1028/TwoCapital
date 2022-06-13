# TwoCapital
Repo for R&amp;D model

## Prerequisite

Install PETSc, Python3.

## Intalling PETSc 

PETSc requires a Linux or Unix based OS. For Windows machine we therefore need to simulate such an OS. For MacOS machines unles they contain an M1 chip (M1 CPU) one can configure (install) the library directly following the instructions provided on the PETSc webpage. If the MacOS machine on which an installation is required contains a CPU in the M1 family please follow the instructions below. 

The basic idea is to create a virtual OS environment either by using the standard (and old) method of creating virtual machines or, alternatively, using docker containers. Virtual machines (VMs) tend to consume more resources compared to docker containers which are much more light-weight and, with the correct set up, are generally faster. Virtual machines are, however, overall more user friendly as the "correct set up" of docker containers requires some technical workarounds. 


### <a name="windows"></a>Installing on Windows

1. Download and install VirtualBox
2. Download an image file for the operating system to be simulated e.g. Ubuntu is a good choice and the latest version (20.04.03 as of Feb 8, 2022) is available [here](https://ubuntu.com/download/desktop).
3. Through virtual box create a virtual machine for a Linux OS. Make sure to leave enough space for the virtual machine i.e. when prompted assign 30GB rather than the default 8GB of disk space and assign at least 8GB (8192 MB) of RAM.
4. In terminal run: 

  `sudo apt-get install build-essential`

  this will install a `C++` compiler and other essentials

5. In terminal run: 

  `sudo apt-get install gfortran`

  this will install a `FORTRAN` compiler.

6. Install `Cython` - `Python` bindings for `C`. This can be done in 2 ways:
    - Installing the `Cython` [package](https://cython.readthedocs.io/en/stable/src/quickstart/install.html) directly with

    `pip install Cython` 
    - Installing `Anaconda` which contains `Cython` within the standard distribution. Installing `Anaconda` has advantages. It's a complete package management system which provide all necessary libraries and/or packages for scientific computing. `Anaconda` does not replace the default package manager `pip` but rather complements it and allows for environment management i.e. creating different workspaces containing different `Python` packages to enable easy testing, backwards compatibility checks, and resolution of package dependency issues. I user friendly installation tutorial on Ubuntu can be found [here](https://phoenixnap.com/kb/how-to-install-anaconda-ubuntu-18-04-or-20-04).

7. Follow the instructions on `Python` bindings [below](#python_bindings). 



### Installing on MacOS (M1 CPU)
The idea is to coerce docker containers to behave like VMs. This can be achieved through different VM management software such as Docker (Docker Destkop) or Vagrant:

For Docker:

1. Download and install Docker Desktop
2. Download an Ubuntu [image](https://hub.docker.com/_/ubuntu)
3. Create an Unbutu container with Docker
4. Follow steps 4 through 7 from the section ['Installing on Windows'](#windows).


For Vagrant:
1. Follow the instructions in this [tutorial](https://medium.com/nerd-for-tech/developing-on-apple-m1-silicon-with-virtual-environments-4f5f0765fd2f).


### <a name="python_bindings"></a> PETSc  with `Python` bindings

To run PETSc with `Python` we also need `Python` bindings provided by the `petsc4py` package. To install `petsc4py` and properly configure PETSc to work with `Python` follow these steps:

1. For maintainance convenience, choose a name for the arch you will be configuring. In the folder where PETSc download folder run the following configuration command:

```
 ./configure --with-cc=gcc --with-cxx=g++ --with-fc=gfortran --download-mpich --download-fblaslapack --with-petsc4py --with-debugging=no PETSC_ARCH=<YourChoiceOfArchName>
```
> **_Note:_** `mpi` is not absolute necessary at the moment. If mpich takes up a long time to download, or it runs into ERROR message, you can choose to disable to download option by changing the flag `--download-mpich` to `--with-mpi=0`


Following the steps suggested by the configuration output:
```
make PETSC_ARCH=<YourChoiceOfArchName> all
```
And complete the make check following the instruction
```
make PETSC_ARCH=<YourChoiceOfArchName> check
```


2. Store environmental variables. A useful step is to check which file are loaded when you start a shell session using the following command:

```
echo exit | strace bash -li |& grep '^open'
```

The output will tell you whether `~/.bashrc` and/or `~/.bash_profile` and/or `~/.profile` are loaded.
Suppose `~/.bashrc` are load when you start a bash session. Use the following command to store the environmental variables.
If multiple files show up, choose any of the files and add the following to the file.
For instance, if `~/.bashrc` shows up, use the following command:

```
  echo 'export PETSC_DIR=</path/to/petsc>' >> ~/.bashrc
  echo 'export PETSC_ARCH=<YourChoiceOfArchName>' >> ~/.bashrc
  echo `export PYTHONPATH=$PETSC_DIR/$PETSC_ARCH/lib` >> ~/.bashrc
```


3. Check if the configuration is successful: 
```
exec $SHELL
echo $PETSC_DIR
echo $PETSC_ARCH
echo $PYTHONPATH
```
to see if the outputs are correct. They should provide you with what you entered in step 2 above.

Enter the following commmand to see if Python packages are successfully installed:
```
pip list | grep petsc4py
pip list | grep petsclinearsystem
```
or
```
pip3 list | grep petsc4py
pip3 list | grep petsclinearsystem
```
to see if `petsc4py` and `petsclinearsystem` has been installed successfully.



## Install the linear solver and other necessary packages

- For UNIX-like OS users, use `install.sh` to install necessary packages:
```
chmod +x install.sh
```
```
./install.sh
```
The file installs `eigen3` solver and `PETSc` solver.
- (Optional) installation of `SolveLinSys`: answer "y" to the question *Do you want to install eigen3 solver?*

## Scripts and Model

## The Project for R&D model are stored under `./tech4D/` folder.

A write-up is under the `./write-ups/write-up.pdf`

In py file, `linearsolver` stands for the linear solver to use: 
available solution:

- `pestsc` for PETSc + C implementation of coefficient matrix
- `petsc4py` for PETSc and numpy sparse matrix
- `eigen3` for `SolveLinSys`



`post-jump-gamma.py` corresponds to section 1.2.1 in `write-up.pdf` with state variables log K, R, Y.
The file is used together with several flags. Enter the following command for more details:
```
python post-jump-gamma.py -h
```
It would print out the following info:
```
usage: post-jump-gamma.py [-h] [--gamma GAMMA] [--eta ETA] [--epsilon EPSILON] [--fraction FRACTION] [--keep-log]

Set damage curvature value, and hyper parameters for the optimization problem.

optional arguments:
  -h, --help           show this help message and exit
  --gamma GAMMA        Index number of gamma_3 in the list of gamma_3 values. By default, we are solving with 10 damage function, then the value could be 0,1,...,9.
  --eta ETA            Value of eta, default = 0.17
  --epsilon EPSILON    Value of epsilon, default = 0.1
  --fraction FRACTION  Value of fraction of control update, default = 0.1
  --keep-log           Flag to keep a log of the computation
```

The default settigs are
```
python post-jump-gamma.py --gamma 0 --eta 0.17 --epsilon 0.1 --fraction 0.1
```

`HJB-4d-logL.py` corresponds to section 2 Pre jump HJB in `write-up.pdf`

## The Project for non-linear carbon and temperature dynamics are stored under `./nonlinearCarbon/` folder.
