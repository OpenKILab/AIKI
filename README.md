# aiki

[![codecov](https://codecov.io/gh/OpenKILab/AIKI/branch/main/graph/badge.svg?token=AIKI_token_here)](https://codecov.io/gh/OpenKILab/AIKI)
[![CI](https://github.com/OpenKILab/AIKI/actions/workflows/main.yml/badge.svg)](https://github.com/OpenKILab/AIKI/actions/workflows/main.yml)



Awesome aiki created by OpenKILab

## Installation
```py
#not supported yet
```

## Pre-requisites

1. [Optinal] Install [conda](https://docs.conda.io/en/latest/miniconda.html) if you want to keep your system clean. 
Then create a conda environment.

```bash
conda create -n aiki python=3.10
conda activate aiki
```
2. Install [poetry](https://python-poetry.org/docs/#installation)

3. Config
```bash
export OPENAI_API_KEY="your_openai_api_key"
```

### hint:
1. export PYTHONPATH


## Usage

### Quick-Start

```python
import os
from aiki.aiki import AIKI

# initialize your lib
a_k_ = AIKI(db_name = "flick")

# add text data
a_k_.index("我上周出去遛狗了")
# add image data with path
image_path = os.path.abspath("resource/source/imgs/外滩小巷.jpg")
a_k_.index(image_path)
# add more ...

# try to find something in the lib
print(a_k_.retrieve("我上周出去遛狗了么", num=2))
print(a_k_.retrieve("几个人在街上，有些人正在使用手机，另外一些人在骑自行车", num=2))
```

## Benchmark
**TBD**

## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.
