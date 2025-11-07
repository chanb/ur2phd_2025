# UR2PhD Project

Below is the "minimal setup"
## Setup
1. Install Python3.10: https://www.python.org/downloads/  
Note that that the installation steps may be different for different OS.

1. Once you have Python3.10 installed, please install `virtualenv` following steps from https://virtualenv.pypa.io/en/latest/installation.html

2. We can now setup a Python virtual environment following the next section

### Setting up virtual environment
```
# Create virtual environment named .venv (one time)
virtualenv -p <PATH/TO/PYTHON_3.10> .venv

# Activate the virtual environment
source .venv/bin/activate

# Install the necessary packages (one time)
pip install numpy==2.1.3
pip install torch==2.5.1
pip install torchvision==0.20.1
pip install matplotlib==3.9.2
pip install jupyter==1.1.1
pip install ipympl==0.9.4
pip install tensorboard
pip install tqdm

# To deactivate the virtual environment
deactivate
```

We will be using `numpy` and `torch` quite extensively.
`jupyter` will be used for quick prototyping as it provides a great interactive interface. Otherwise one can use [Visual Studio Code](https://code.visualstudio.com/) or [Google Colab](https://colab.research.google.com/) to run `.ipynb` files.