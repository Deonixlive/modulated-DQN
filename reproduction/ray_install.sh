#!/usr/bin/bash

conda create -y -n test python==3.10

# https://www.tensorflow.org/install/pip
conda install -n test -y -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0
conda run -n test export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

conda run -n test pip install ray[rllib]==2.3.0 tensorflow==2.11.0

conda run -n test python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

#Es gibt neuere Versionen, sind aber nicht kompatibel mit der Version
#die Rllib für Tensorflow braucht.
conda run -n test pip install tensorflow_probability==0.18

conda run -n test pip install gymnasium[atari]==0.26.3
conda run -n test pip install gym[atari]==0.26.2 gym[accept-rom-license] atari_py
conda run -n test pip install gymnasium[accept-rom-license]==0.26.3

#zum Daten visualisieren
#conda run -n test pip install tensorboard
conda run -n test pip install jupyterlab

