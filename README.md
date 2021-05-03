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

`apt-get install libsndfile1`
## Methods

The source code is mostly taken from https://github.com/WeidiXie/VGG-Speaker-Recognition (python 2), which I re-wrote in python 3 with better documentation and some changes in model.


## How code is organized?

