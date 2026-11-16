# Biocomposite Surrogate Training

Train a probabilistic biocomposite stiffness surrogate using generated datasets.

Datasets and trained weights are not included. Run all commands from the
repository root, where `trainFiberNetwork.py` and `requirements.txt` are located.

## Conda Environment

Install Conda using the [official installation instructions](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html)
if needed, then open a new terminal. On Linux with Bash, run `conda init bash`
and restart the terminal if your shell has not been initialized for Conda.

Create a dedicated training environment:

```sh
conda create --name biocomp-training --override-channels --channel conda-forge python=3.10 pip
conda activate biocomp-training
python -m pip install -r requirements.txt
python -m pip check
```

Verify the imports:

```sh
python -c "import sys, tensorflow as tf, keras, tensorflow_probability as tfp; import dataLoaders, fiberNet; print(sys.executable); print(tf.__version__, keras.__version__, tfp.__version__)"
```

The environment pins TensorFlow 2.10.0, Keras 2.10.0, and TensorFlow Probability
0.18.0 for compatibility with the training code. Keep it separate from the
dataset generator's environment, which uses different NumPy dependencies. The bundled `tfkan`
implementation is imported locally; do not install a replacement from PyPI.

Activate `biocomp-training` again in each new terminal. CPU training is
supported and does not require MATLAB or MATLAB Runtime. GPU use requires
a separately configured CUDA/cuDNN installation compatible with TensorFlow
2.10; GPU setup is not included here.

## Training Data

Use a MATLAB v7.3 dataset produced by the biocomposite stiffness dataset
generator. Place it at `data/training.mat`, or pass its full path to training.

The file contains `x` and `y` with the following dimensions:

| Variable | MATLAB | h5py | Contents |
| --- | --- | --- | --- |
| `x` | N-by-18 | 18-by-N | Training inputs before parameter scaling |
| `y` | N-by-5 | 5-by-N | `E_T`, `E_L`, `nu_T`, `nu_LT`, `mu_LT` |

The loader subtracts 0.5 from inputs, returns inputs with shape
`(N, 18, 1)`, and takes natural logarithms of the three moduli in output
columns 1, 2, and 5. Do not transpose, center, or logarithmically transform
the dataset beforehand. Moduli must be positive, and the dataset must contain
enough accepted samples for both training and validation.

## Train

```sh
python trainFiberNetwork.py data/training.mat 100 256 0.2
```

The positional arguments are the dataset path, maximum number of epochs,
batch size, and validation fraction. Run from the repository root; results
are saved in `kan-models/`.

The script trains PBKAN with layer size 24, a shuffled training/validation
split, negative log-likelihood loss, a learning-rate schedule,
early stopping, and the KAN grid-update callback. The model architecture
and layer size are defined in the source code.

For a short run on a small dataset:

```sh
python trainFiberNetwork.py data/training.mat 6 8 0.25
```

Results are written to:

- `kan-models/fiber-model-<timestamp>`: HDF5 weight arrays named `weight0`,
  `weight1`, and so on; this is a weights file, not a serialized full model.
- `kan-models/fiber-model-history-<timestamp>.json`: training and validation
  history.

Datasets, saved weights, histories, and Python caches are ignored by Git.
After training, leave the environment with `conda deactivate`.
