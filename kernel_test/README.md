# private_tst
Python code (tested on 2.7) for differentially private two-sample test, the details are described in the following paper: 

Anant Raj*, Ho Chung Leon Law*, Dino Sejdinovic, Mijung Park, __A Differentially Private Kernel Two-Sample Test__ [arxiv](https://arxiv.org/abs/1808.00380) 

\* denotes equal contribution

## Setup
To setup as a package, clone the repository and run
```
python setup.py develop
```

## Structure
The directory is organised as follows:
* __private_me__: contains the main code, with the main API found in train_test.py

The following will show the available dataset options:
```
python train_test.py --h
```
After selection of an dataset, one can show the options through (say for same_gaussian):
```
python train_test.py same_gaussian --h
```
The following would run a SCF test with NTE setting (specified by --privacy-type local_meanCov) using the approximate private null distribution. The results will be saved to save.pkl .
```
python train_test.py same_gaussian --test-type SCF --privacy-type local_meanCov --null private save.pkl
```
To produce the experiments in the paper, we use the default settings in the API scripts and we vary test type (--test-type), different privacy approaches (--privacy-type), epsilon (--epsilon), delta (--delta) and null distribution (--null). 

We give special thanks to Wittawat Jitkrittum, whose two-sample test [github](https://github.com/wittawatj/interpretable-test) package we build on.

