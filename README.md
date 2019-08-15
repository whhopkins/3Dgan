# 3Dgan
3Dgan implementation using keras2 with tensorflow backend can be found in the keras folder. 

AngleArch3dGAN.py is the architecture and AngleTrain3dGAN.py is the training script. 

The weights dir is used to store weights from training. If weights for different trainings are to be saved then --name can be used at command line to identify the training.

analysis/LossPlotsPython.py : Plots Train and Test losses. If training has been named different from default then at command line --historyfile and --outdir must be used to specify the location of loss history generated from a training and the dir where results should be stored respectively.

analysis/RootAnalysis.py : Physics Evaluation results. If training has been named different from default then paths for weights(--dweights & --gweights) and output(--plotdir) must be provided.

Tested with:
keras: 2.2.4
tensorflow : 1.14.1

Analysis scripts also uses Root and matplotlib

## Installation of packages on dgx2

1. Install a local copy of python3 on dxg (or whatever machine you plan on running on)
   a. Download python 3.6: wget https://www.python.org/ftp/python/3.6.8/Python-3.6.8.tgz
   b. Untar: tar zxvf Python-3.6.8.tgz; cd Python-3.6.8
   c. Make an installation directory: mkdir ~/opt0/
   d. Configure/make/make install: ./configure --prefix=$HOME/opt2/python-3.6.8 --enable-optimizations; make; make install
   e. Add your installation to your PATH: export PATH=$HOME/opt2/python-3.6.8/bin:$PATH

2. Install the libraries you need
   pip3 install tensorflow-gpu keras matplotlib scikit-learn memory_profiler --user

3. Run the command for training
   python3 AngleTrain3dGAN.py --datapath "/home/whopkins/data_3dgan/*" --nEvents 1000 --nbepoch 2