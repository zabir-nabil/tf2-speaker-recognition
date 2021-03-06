# tf2-speaker-recognition


## Installation

Install docker, nvidia-docker.

### Docker container

#### CUDA 9.0
**N.B:** If you have an old NVIDIA driver like me (which I can't change :disappointed: as some other docker containers are running which depends on that version), you may need to build tensorflow from source. I have used a version for cuda 9 which is available here: [tensorflow 2.1 python 3.7 cuda 9.0 wheel](https://drive.google.com/file/d/1JtxGVpJQAIRxEzdIyIQsGY0axU0a0ISo/view)

```
# Could not load dynamic library 'libcusolver.so.9.0'; undefined symbol: GOMP_critical_end;
import ctypes
ctypes.CDLL("libgomp.so.1", mode=ctypes.RTLD_GLOBAL)
```

```
apt-get install libav-tools # if you want to read .m4a files
```

`pip install gdown`

`gdown https://drive.google.com/uc?id=1JtxGVpJQAIRxEzdIyIQsGY0axU0a0ISo`

#### Using docker container

`docker pull nvidia/cuda:9.0-cudnn7-devel`

`nvidia-docker run -it -v path_in_drive:mapping_in_docker --net=host nvidia/cuda:9.0-cudnn7-devel bash`

cd into mapped folder with git.

`apt install unzip; unzip tensorflow-2.1.0-cp37-cp37m-linux_x86_64.whl.zip; apt install wget`

`bash install_conda.sh`

`source ~/.bashrc; conda update conda`

##### conda environment

`conda create -n tf2sr python==3.7.5`

`conda activate tf2sr`

`pip install -r reqs.txt` 

`pip uninstall h5py; pip install 'h5py<3.0.0'` # old bug

`apt-get install libsndfile1`

## Methods

The source code is taken from https://github.com/WeidiXie/VGG-Speaker-Recognition (python 2), which was modified to work in python3.


## How code is organized?

To train, `python -W ignore train.py` # set the params inside

`python -W ignore train.py --resume baseline_weights/weights.h5`

