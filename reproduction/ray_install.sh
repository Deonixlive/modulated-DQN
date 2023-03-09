#!/usr/bin/bash

conda create -n test python==3.10

conda activate test

# https://www.tensorflow.org/install/pip
conda install -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

pip install ray[rllib]==2.3.0 tensorflow==2.11.0

python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

#Es gibt neuere Versionen, sind aber nicht kompatibel mit der Version
#die Rllib für Tensorflow braucht.
pip install tensorflow_probability==0.18

pip install gym[atari]==0.26.2 gym[accept-rom-license] atari_py
pip install gymnasium[atari]==0.26.3
pip install gymnasium[accept-rom-license]==0.26.3

#zum Daten visualisieren
pip install tensorboard
#pip install jupyterlab