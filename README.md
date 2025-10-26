## Install Miniconda and create environment for Ubuntu

# download miniconda (if not installed)
> wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh

### (Optional, if needed) start -->>
# follow prompts, then restart shell or `source ~/.bashrc`

# Conda now requires you to accept the Terms of Service (ToS) before using its default channels. You just need to accept them.

# Run these commands one by one:
> conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
> conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# or Make it permanent
# To avoid doing this every time, add it to your .bashrc:

> echo 'export PATH="/home/bulipe/miniconda3/bin:$PATH"' >> ~/.bashrc
> source ~/.bashrc
### (optional) <<-- end 

# create env and activate
> conda create -n venv_name python=3.x -y
> conda activate venv_name

# upgrade pip and install common libs
> pip install --upgrade pip 
> pip install numpy pandas matplotlib scikit-learn pillow opencv-python

# Install TensorFlow (CPU build):
> pip install tensorflow   # CPU version on non-NVIDIA machines

# Locally run train command
> python train_local.py

# Locally run inference command
> python inference_local.py


### Note: After added bellow those command for jupyter notebook

## Set up environment for jupyter Notebook
# Install Jupyter and TensorFlow (if not already)
> pip install --upgrade pip
> pip install jupyter tensorflow matplotlib scikit-learn

# Then launch Jupyter Notebook run command:
> jupyter notebook

# Happy coding for ML ):

