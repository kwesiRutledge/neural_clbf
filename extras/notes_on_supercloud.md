# Notes on Supercloud

## Introduction

## Logging in to Supercloud
You'll use your ssh keys and a custom ssh to log in to a "login"" node in Supercloud.
(You can observe the commands you need to do this in `login_to_supercloud.sh`) in this directory.

## Setting up your environment

### Loading Modules
You'll need to load a few modules before you get started.

Use the following commands to load modules:
```bash
module load module_name
```
Some suggested modules to load are Anaconda and Gurobi.
(They have different names in the system along with version numbers.)

You can see which modules are currently loaded with:
```bash
module list
```

And you can observe what modules are available with the command
```bash
module avail
```

### Starting Conda

The command `conda activate` will not work on SuperCloud. Instead, you'll need to use the following command:
```bash
source activate
```

### Installing via Pip

The following commands are recommended for installation via pip. 
```bash
mkdir /state/partition1/user/$USER
export TMPDIR=/state/partition1/user/$USER
pip install --user --no-cache-dir packageName
```

This appears to set up a temporary directory for pip to use that has larger space limitations than normal.

Occasionally, you may need to reinstall a package via pip from a specific version. For this situation, use this:
```bash
pip install --user --no-cache-dir --force-reinstall -v packageName==versionNumber
```