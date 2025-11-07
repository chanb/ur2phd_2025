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

# To deactivate the virtual environment
deactivate
```

We will be using `numpy` and `torch` quite extensively.
`jupyter` will be used for quick prototyping as it provides a great interactive interface. Otherwise one can use [Visual Studio Code](https://code.visualstudio.com/) or [Google Colab](https://colab.research.google.com/) to run `.ipynb` files.

## Collaborative project
### Installing additional packages for the project
You have to install the two extra packages:
```
pip install tensorboard
pip install tqdm
```

### Running the project
To quickly run the dummy code
```
# Assuming $PWD is the repository location
python project/src/main.py --config_path=project/configs/dummy.json
```
Here, all training experiments should go through `main.py` which generates all the artifacts including learning curves and checkpoints.
The argument `--config_path` corresponds to the experiment configuration, where you provide the model hyperparameters, dataset configurations, training hyperparameters, etc.
A directory will be created under the path specified by the experiment configuration `<logging_config/save_path>`. The directory will be named as `<logging_config/experiment_name>-mm-dd-yy_HH_mm_ss-<experiment_id>`, where `<experiment_id>` is automatically-generated unique ID (using `uuid`).

### Project code structure
Our project is as follows:
```
.
├── configs/
├── src/
│   ├── datasets/
│   ├── models/
│   ├── learners/
│   ├── constants.py
│   ├── main.py
│   ├── train.py
│   └── utils.py
└── vis/
```
- `configs/` contains a list of JSON files, specifying the experiment configurations.
- `vis/` contains a list of visualization tools of trained models---this can be visualizing heatmaps, attention score, relevance, gradient, token embeddings, etc.
- `src/` contains the training code.

To delve deeper into the training code, we have the following:
- `datasets` contains all the datasets, in our case they will be addition, subtraction, multiplication, etc.
- `models` contains the model architectures, we would want something like variants of a transformer implemented here.
- `learners` contains the learning algorithms, we want to implement a learner for next-token prediction in this case---this learner will sample batches of data, update the parameters of the model, etc.
- `main.py` is the main entrypoint, responsible for creating the logging directories and checkpoints---think of it as an orchestrator that sets up the experiment workspace.
- `train.py` is the code that actually runs the learner, orchestrating when training steps and validation steps are performed. Essentially, it instantiates a learner from `learners`, runs that learner, and keeps track of the training progress.
- `constants.py` contains all the magic constants---we currently use it to assign strings for logging.
- `utils.py` contains some helper functions that are currently used by other files.
The above four Python files are unlikely to be changed, maybe except for `constants.py`.

I have included some example Python files under `datasets`, `models`, and `learners`:
- A `Dataset` should be an iterator that yields a dictionary, containing at least: `input` and `target`
- Likewise, a `Model` should be a `torch.nn.Module` that outputs a dictionary, containing at least: `output` when `forward` is called. Note that we're using a dictionary because if we want to compute the attention score, for example, we can include it here.
- The `NextTokenLearner` is missing the `train_step`, `update`, and `validaion` methods---I have included docstring to describe each method.

**NOTE 1:** You have full freedom to modify this codebase, so long as everyone is aware of the modification (do this through a Pull Request and we will review it).
**NOTE 2:** The choices of the model architecture, learning objective, datasets, and metrics are open to discussion---I suggest asking for forgiveness after (it usually works out).