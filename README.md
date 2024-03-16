# qwen_1_8chat_finetune
 基于lora微调Qwen1.8chat的实战教程

- 日期：2024-3-16
- 作者：小知
- 基于lora参数微调Qwen1.8chat模型。

## 1.环境配置

前提: 已经配置好 `GPU` 环境。

- GPU：NVIDIA A10 cuda 11.8
- tensorflow==2.14


```python
# 查看GPU
!nvidia-smi
```

    Sat Mar 16 16:30:47 2024       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 470.82.01    Driver Version: 470.82.01    CUDA Version: 11.8     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  NVIDIA A10          Off  | 00000000:00:08.0 Off |                    0 |
    |  0%   27C    P8     8W / 150W |      0MiB / 22731MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+



```python
#!pip install -r requirements_qwen_1_8.txt -i https://mirrors.aliyun.com/pypi/simple
```


```python
!pip install deepspeed transformers==4.32.0 peft pydantic==1.10.13 transformers_stream_generator einops tiktoken modelscope
```

    Looking in indexes: https://mirrors.aliyun.com/pypi/simple
    Requirement already satisfied: deepspeed in /opt/conda/lib/python3.10/site-packages (0.12.3)
    Requirement already satisfied: transformers==4.32.0 in /opt/conda/lib/python3.10/site-packages (4.32.0)
    Requirement already satisfied: peft in /opt/conda/lib/python3.10/site-packages (0.6.2)
    Requirement already satisfied: pydantic==1.10.13 in /opt/conda/lib/python3.10/site-packages (1.10.13)
    Requirement already satisfied: transformers_stream_generator in /opt/conda/lib/python3.10/site-packages (0.0.4)
    Requirement already satisfied: einops in /opt/conda/lib/python3.10/site-packages (0.7.0)
    Requirement already satisfied: tiktoken in /opt/conda/lib/python3.10/site-packages (0.5.1)
    Requirement already satisfied: modelscope in /opt/conda/lib/python3.10/site-packages (1.10.0)
    Requirement already satisfied: filelock in /opt/conda/lib/python3.10/site-packages (from transformers==4.32.0) (3.13.1)
    Requirement already satisfied: huggingface-hub<1.0,>=0.15.1 in /opt/conda/lib/python3.10/site-packages (from transformers==4.32.0) (0.19.4)
    Requirement already satisfied: numpy>=1.17 in /opt/conda/lib/python3.10/site-packages (from transformers==4.32.0) (1.26.1)
    Requirement already satisfied: packaging>=20.0 in /opt/conda/lib/python3.10/site-packages (from transformers==4.32.0) (23.1)
    Requirement already satisfied: pyyaml>=5.1 in /opt/conda/lib/python3.10/site-packages (from transformers==4.32.0) (6.0.1)
    Requirement already satisfied: regex!=2019.12.17 in /opt/conda/lib/python3.10/site-packages (from transformers==4.32.0) (2023.10.3)
    Requirement already satisfied: requests in /opt/conda/lib/python3.10/site-packages (from transformers==4.32.0) (2.31.0)
    Requirement already satisfied: tokenizers!=0.11.3,<0.14,>=0.11.1 in /opt/conda/lib/python3.10/site-packages (from transformers==4.32.0) (0.13.3)
    Requirement already satisfied: safetensors>=0.3.1 in /opt/conda/lib/python3.10/site-packages (from transformers==4.32.0) (0.4.0)
    Requirement already satisfied: tqdm>=4.27 in /opt/conda/lib/python3.10/site-packages (from transformers==4.32.0) (4.65.0)
    Requirement already satisfied: typing-extensions>=4.2.0 in /opt/conda/lib/python3.10/site-packages (from pydantic==1.10.13) (4.8.0)
    Requirement already satisfied: hjson in /opt/conda/lib/python3.10/site-packages (from deepspeed) (3.1.0)
    Requirement already satisfied: ninja in /opt/conda/lib/python3.10/site-packages (from deepspeed) (1.11.1.1)
    Requirement already satisfied: psutil in /opt/conda/lib/python3.10/site-packages (from deepspeed) (5.9.6)
    Requirement already satisfied: py-cpuinfo in /opt/conda/lib/python3.10/site-packages (from deepspeed) (9.0.0)
    Requirement already satisfied: pynvml in /opt/conda/lib/python3.10/site-packages (from deepspeed) (11.5.0)
    Requirement already satisfied: torch in /opt/conda/lib/python3.10/site-packages (from deepspeed) (2.1.0+cu118)
    Requirement already satisfied: accelerate>=0.21.0 in /opt/conda/lib/python3.10/site-packages (from peft) (0.24.1)
    Requirement already satisfied: addict in /opt/conda/lib/python3.10/site-packages (from modelscope) (2.4.0)
    Requirement already satisfied: attrs in /opt/conda/lib/python3.10/site-packages (from modelscope) (23.1.0)
    Requirement already satisfied: datasets>=2.14.5 in /opt/conda/lib/python3.10/site-packages (from modelscope) (2.15.0)
    Requirement already satisfied: gast>=0.2.2 in /opt/conda/lib/python3.10/site-packages (from modelscope) (0.5.4)
    Requirement already satisfied: oss2 in /opt/conda/lib/python3.10/site-packages (from modelscope) (2.18.3)
    Requirement already satisfied: pandas in /opt/conda/lib/python3.10/site-packages (from modelscope) (2.1.3)
    Requirement already satisfied: Pillow>=6.2.0 in /opt/conda/lib/python3.10/site-packages (from modelscope) (10.1.0)
    Requirement already satisfied: pyarrow!=9.0.0,>=6.0.0 in /opt/conda/lib/python3.10/site-packages (from modelscope) (14.0.1)
    Requirement already satisfied: python-dateutil>=2.1 in /opt/conda/lib/python3.10/site-packages (from modelscope) (2.8.2)
    Requirement already satisfied: scipy in /opt/conda/lib/python3.10/site-packages (from modelscope) (1.11.3)
    Requirement already satisfied: setuptools in /opt/conda/lib/python3.10/site-packages (from modelscope) (68.0.0)
    Requirement already satisfied: simplejson>=3.3.0 in /opt/conda/lib/python3.10/site-packages (from modelscope) (3.19.2)
    Requirement already satisfied: sortedcontainers>=1.5.9 in /opt/conda/lib/python3.10/site-packages (from modelscope) (2.4.0)
    Requirement already satisfied: urllib3>=1.26 in /opt/conda/lib/python3.10/site-packages (from modelscope) (1.26.16)
    Requirement already satisfied: yapf in /opt/conda/lib/python3.10/site-packages (from modelscope) (0.30.0)
    Requirement already satisfied: pyarrow-hotfix in /opt/conda/lib/python3.10/site-packages (from datasets>=2.14.5->modelscope) (0.6)
    Requirement already satisfied: dill<0.3.8,>=0.3.0 in /opt/conda/lib/python3.10/site-packages (from datasets>=2.14.5->modelscope) (0.3.6)
    Requirement already satisfied: xxhash in /opt/conda/lib/python3.10/site-packages (from datasets>=2.14.5->modelscope) (3.4.1)
    Requirement already satisfied: multiprocess in /opt/conda/lib/python3.10/site-packages (from datasets>=2.14.5->modelscope) (0.70.14)
    Requirement already satisfied: fsspec<=2023.10.0,>=2023.1.0 in /opt/conda/lib/python3.10/site-packages (from fsspec[http]<=2023.10.0,>=2023.1.0->datasets>=2.14.5->modelscope) (2023.10.0)
    Requirement already satisfied: aiohttp in /opt/conda/lib/python3.10/site-packages (from datasets>=2.14.5->modelscope) (3.9.1)
    Requirement already satisfied: six>=1.5 in /opt/conda/lib/python3.10/site-packages (from python-dateutil>=2.1->modelscope) (1.16.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /opt/conda/lib/python3.10/site-packages (from requests->transformers==4.32.0) (2.0.4)
    Requirement already satisfied: idna<4,>=2.5 in /opt/conda/lib/python3.10/site-packages (from requests->transformers==4.32.0) (3.4)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.10/site-packages (from requests->transformers==4.32.0) (2023.7.22)
    Requirement already satisfied: sympy in /opt/conda/lib/python3.10/site-packages (from torch->deepspeed) (1.12)
    Requirement already satisfied: networkx in /opt/conda/lib/python3.10/site-packages (from torch->deepspeed) (3.2.1)
    Requirement already satisfied: jinja2 in /opt/conda/lib/python3.10/site-packages (from torch->deepspeed) (3.1.2)
    Requirement already satisfied: triton==2.1.0 in /opt/conda/lib/python3.10/site-packages (from torch->deepspeed) (2.1.0)
    Requirement already satisfied: crcmod>=1.7 in /opt/conda/lib/python3.10/site-packages (from oss2->modelscope) (1.7)
    Requirement already satisfied: pycryptodome>=3.4.7 in /opt/conda/lib/python3.10/site-packages (from oss2->modelscope) (3.19.0)
    Requirement already satisfied: aliyun-python-sdk-kms>=2.4.1 in /opt/conda/lib/python3.10/site-packages (from oss2->modelscope) (2.16.2)
    Requirement already satisfied: aliyun-python-sdk-core>=2.13.12 in /opt/conda/lib/python3.10/site-packages (from oss2->modelscope) (2.14.0)
    Requirement already satisfied: pytz>=2020.1 in /opt/conda/lib/python3.10/site-packages (from pandas->modelscope) (2023.3.post1)
    Requirement already satisfied: tzdata>=2022.1 in /opt/conda/lib/python3.10/site-packages (from pandas->modelscope) (2023.3)
    Requirement already satisfied: jmespath<1.0.0,>=0.9.3 in /opt/conda/lib/python3.10/site-packages (from aliyun-python-sdk-core>=2.13.12->oss2->modelscope) (0.10.0)
    Requirement already satisfied: cryptography>=2.6.0 in /opt/conda/lib/python3.10/site-packages (from aliyun-python-sdk-core>=2.13.12->oss2->modelscope) (41.0.3)
    Requirement already satisfied: multidict<7.0,>=4.5 in /opt/conda/lib/python3.10/site-packages (from aiohttp->datasets>=2.14.5->modelscope) (6.0.4)
    Requirement already satisfied: yarl<2.0,>=1.0 in /opt/conda/lib/python3.10/site-packages (from aiohttp->datasets>=2.14.5->modelscope) (1.9.3)
    Requirement already satisfied: frozenlist>=1.1.1 in /opt/conda/lib/python3.10/site-packages (from aiohttp->datasets>=2.14.5->modelscope) (1.4.0)
    Requirement already satisfied: aiosignal>=1.1.2 in /opt/conda/lib/python3.10/site-packages (from aiohttp->datasets>=2.14.5->modelscope) (1.3.1)
    Requirement already satisfied: async-timeout<5.0,>=4.0 in /opt/conda/lib/python3.10/site-packages (from aiohttp->datasets>=2.14.5->modelscope) (4.0.3)
    Requirement already satisfied: MarkupSafe>=2.0 in /opt/conda/lib/python3.10/site-packages (from jinja2->torch->deepspeed) (2.1.3)
    Requirement already satisfied: mpmath>=0.19 in /opt/conda/lib/python3.10/site-packages (from sympy->torch->deepspeed) (1.3.0)
    Requirement already satisfied: cffi>=1.12 in /opt/conda/lib/python3.10/site-packages (from cryptography>=2.6.0->aliyun-python-sdk-core>=2.13.12->oss2->modelscope) (1.15.1)
    Requirement already satisfied: pycparser in /opt/conda/lib/python3.10/site-packages (from cffi>=1.12->cryptography>=2.6.0->aliyun-python-sdk-core>=2.13.12->oss2->modelscope) (2.21)
    [33mDEPRECATION: omegaconf 2.0.6 has a non-standard dependency specifier PyYAML>=5.1.*. pip 24.0 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of omegaconf or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063[0m[33m
    [0m[33mDEPRECATION: pytorch-lightning 1.7.7 has a non-standard dependency specifier torch>=1.9.*. pip 24.0 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063[0m[33m
    [0m[33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m23.3.1[0m[39;49m -> [0m[32;49m24.0[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m


## 2.模型下载

阿里魔搭社区notebook的[jupyterLab](https://www.modelscope.cn/my/mynotebook/authorization)里：下载模型会缓存在 `/mnt/workspace/.cache/modelscope/`。一般会缓存到你的C盘或用户空间，所以要根据自己情况查看模型。也可以通过下面日志查看模型所在位置，如`2024-03-16 16:30:54,106 - modelscope - INFO - Loading ast index from /mnt/workspace/.cache/modelscope/ast_indexer`。


```python
%%time
from modelscope import snapshot_download
model_dir = snapshot_download('qwen/Qwen-1_8B-Chat')
!ls /mnt/workspace/.cache/modelscope/qwen/Qwen-1_8B-Chat/
```

    2024-03-16 16:30:54,103 - modelscope - INFO - PyTorch version 2.1.0+cu118 Found.
    2024-03-16 16:30:54,106 - modelscope - INFO - TensorFlow version 2.14.0 Found.
    2024-03-16 16:30:54,106 - modelscope - INFO - Loading ast index from /mnt/workspace/.cache/modelscope/ast_indexer
    2024-03-16 16:30:54,447 - modelscope - INFO - Loading done! Current index file version is 1.10.0, with md5 44f0b88effe82ceea94a98cf99709694 and a total number of 946 components indexed
    /opt/conda/lib/python3.10/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm
    2024-03-16 16:30:56,478 - modelscope - WARNING - Model revision not specified, use revision: v1.0.0
    Downloading: 100%|██████████| 8.21k/8.21k [00:00<00:00, 48.8MB/s]
    Downloading: 100%|██████████| 50.8k/50.8k [00:00<00:00, 146MB/s]
    Downloading: 100%|██████████| 244k/244k [00:00<00:00, 41.7MB/s]
    Downloading: 100%|██████████| 135k/135k [00:00<00:00, 13.3MB/s]
    Downloading: 100%|██████████| 910/910 [00:00<00:00, 9.04MB/s]
    Downloading: 100%|██████████| 77.0/77.0 [00:00<00:00, 742kB/s]
    Downloading: 100%|██████████| 2.29k/2.29k [00:00<00:00, 22.2MB/s]
    Downloading: 100%|██████████| 1.88k/1.88k [00:00<00:00, 21.4MB/s]
    Downloading: 100%|██████████| 249/249 [00:00<00:00, 2.34MB/s]
    Downloading: 100%|██████████| 1.63M/1.63M [00:00<00:00, 22.2MB/s]
    Downloading: 100%|██████████| 1.84M/1.84M [00:00<00:00, 25.6MB/s]
    Downloading: 100%|██████████| 2.64M/2.64M [00:00<00:00, 35.0MB/s]
    Downloading: 100%|██████████| 7.11k/7.11k [00:00<00:00, 7.46MB/s]
    Downloading: 100%|██████████| 80.8k/80.8k [00:00<00:00, 17.7MB/s]
    Downloading: 100%|██████████| 80.8k/80.8k [00:00<00:00, 17.6MB/s]
    Downloading: 100%|█████████▉| 1.90G/1.90G [00:06<00:00, 309MB/s]
    Downloading: 100%|█████████▉| 1.52G/1.52G [00:06<00:00, 238MB/s]
    Downloading: 100%|██████████| 14.4k/14.4k [00:00<00:00, 48.9MB/s]
    Downloading: 100%|██████████| 54.3k/54.3k [00:00<00:00, 48.8MB/s]
    Downloading: 100%|██████████| 15.0k/15.0k [00:00<00:00, 65.4MB/s]
    Downloading: 100%|██████████| 237k/237k [00:00<00:00, 41.3MB/s]
    Downloading: 100%|██████████| 116k/116k [00:00<00:00, 19.4MB/s]
    Downloading: 100%|██████████| 2.44M/2.44M [00:00<00:00, 28.1MB/s]
    Downloading: 100%|██████████| 473k/473k [00:00<00:00, 16.3MB/s]
    Downloading: 100%|██████████| 14.3k/14.3k [00:00<00:00, 60.3MB/s]
    Downloading: 100%|██████████| 79.0k/79.0k [00:00<00:00, 60.0MB/s]
    Downloading: 100%|██████████| 46.4k/46.4k [00:00<00:00, 14.7MB/s]
    Downloading: 100%|██████████| 0.98M/0.98M [00:00<00:00, 42.7MB/s]
    Downloading: 100%|██████████| 205k/205k [00:00<00:00, 55.9MB/s]
    Downloading: 100%|██████████| 19.4k/19.4k [00:00<00:00, 16.7MB/s]
    Downloading: 100%|██████████| 302k/302k [00:00<00:00, 61.5MB/s]
    Downloading: 100%|██████████| 615k/615k [00:00<00:00, 20.1MB/s]
    Downloading: 100%|██████████| 376k/376k [00:00<00:00, 15.2MB/s]
    Downloading: 100%|██████████| 445k/445k [00:00<00:00, 16.1MB/s]
    Downloading: 100%|██████████| 25.9k/25.9k [00:00<00:00, 76.6MB/s]
    Downloading: 100%|██████████| 395k/395k [00:00<00:00, 17.3MB/s]
    Downloading: 100%|██████████| 176k/176k [00:00<00:00, 13.9MB/s]
    Downloading: 100%|██████████| 182k/182k [00:00<00:00, 106MB/s]
    Downloading: 100%|██████████| 824k/824k [00:00<00:00, 6.97MB/s]
    Downloading: 100%|██████████| 426k/426k [00:00<00:00, 18.1MB/s]
    Downloading: 100%|██████████| 433k/433k [00:00<00:00, 66.5MB/s]
    Downloading: 100%|██████████| 466k/466k [00:00<00:00, 16.4MB/s]
    Downloading: 100%|██████████| 403k/403k [00:00<00:00, 75.3MB/s]
    Downloading: 100%|██████████| 9.39k/9.39k [00:00<00:00, 37.0MB/s]
    Downloading: 100%|██████████| 403k/403k [00:00<00:00, 82.7MB/s]
    Downloading: 100%|██████████| 79.0k/79.0k [00:00<00:00, 49.3MB/s]
    Downloading: 100%|██████████| 173/173 [00:00<00:00, 2.15MB/s]
    Downloading: 100%|██████████| 41.9k/41.9k [00:00<00:00, 11.8MB/s]
    Downloading: 100%|██████████| 230k/230k [00:00<00:00, 30.7MB/s]
    Downloading: 100%|██████████| 1.27M/1.27M [00:00<00:00, 151MB/s]
    Downloading: 100%|██████████| 664k/664k [00:00<00:00, 55.4MB/s]
    Downloading: 100%|██████████| 404k/404k [00:00<00:00, 76.9MB/s]

    assets				   model-00002-of-00002.safetensors
    cache_autogptq_cuda_256.cpp	   modeling_qwen.py
    cache_autogptq_cuda_kernel_256.cu  model.safetensors.index.json
    config.json			   NOTICE.md
    configuration.json		   qwen_generation_utils.py
    configuration_qwen.py		   qwen.tiktoken
    cpp_kernels.py			   README.md
    generation_config.json		   tokenization_qwen.py
    LICENSE.md			   tokenizer_config.json
    model-00001-of-00002.safetensors


    


    CPU times: user 14.6 s, sys: 8.76 s, total: 23.4 s
    Wall time: 51.4 s


## 3.本地模型部署


```python
%%time
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig 


query = "识别以下句子中的地址信息，并按照{address:['地址']}的格式返回。如果没有地址，返回{address:[]}。句子为：在一本关于人文的杂志中，我们发现了一篇介绍北京市海淀区科学院南路76号社区服务中心一层的文章，文章深入探讨了该地点的人文历史背景以及其对于当地居民的影响。"
local_model_path = "/mnt/workspace/.cache/modelscope/qwen/Qwen-1_8B-Chat/"
tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(local_model_path, device_map="auto", trust_remote_code=True).eval()
response, history = model.chat(tokenizer, query, history=None)
print("回答如下:\n", response)

```

    The model is automatically converting to bf16 for faster inference. If you want to disable the automatic precision, please manually add bf16/fp16/fp32=True to "AutoModelForCausalLM.from_pretrained".
    Try importing flash-attention for faster inference...
    Warning: import flash_attn rms_norm fail, please install FlashAttention layer_norm to get higher efficiency https://github.com/Dao-AILab/flash-attention/tree/main/csrc/layer_norm
    Loading checkpoint shards: 100%|██████████| 2/2 [00:00<00:00,  2.11it/s]


    回答如下:
     在这个句子中，有三个地址信息：
    1. 北京市海淀区科学院南路76号社区服务中心一层。
    2. 文章深入探讨了该地点的人文历史背景以及其对于当地居民的影响。
    
    按照{address:['地址']}的格式返回：
    在一本关于人文的杂志中，我们发现了一篇介绍北京市海淀区科学院南路76号社区服务中心一层的文章，文章深入探讨了该地点的人文历史背景以及其对于当地居民的影响。
    CPU times: user 3.51 s, sys: 280 ms, total: 3.79 s
    Wall time: 3.79 s


## 4.下载Qwen仓库

克隆Qwen项目，调用`finetune.py`文件进行微调。


```python
%%time
!git clone https://gitcode.com/QwenLM/Qwen.git
```

    正克隆到 'Qwen'...
    remote: Enumerating objects: 1458, done.[K
    remote: Total 1458 (delta 0), reused 0 (delta 0), pack-reused 1458[K
    接收对象中: 100% (1458/1458), 35.31 MiB | 44.42 MiB/s, 完成.
    处理 delta 中: 100% (855/855), 完成.
    CPU times: user 19.4 ms, sys: 22.8 ms, total: 42.3 ms
    Wall time: 1.82 s


## 5.微调与配置

微调脚本能够帮你实现：

- 全参数微调
- LoRA
- Q-LoRA


本次使用 `LoRA` 参数进行微调，调用`Qwen/finetune.py`文件进行配置与微调。


- --model_name_or_path Qwen-1_8B-Chat：指定预训练模型的名称或路径，这里是使用名为"Qwen-1_8B-Chat"的预训练模型。
- --data_path chat.json：指定训练数据和验证数据的路径，这里是使用名为"chat.json"的文件。
- --fp16 True：指定是否使用半精度浮点数（float16）进行训练，这里设置为True。
- --output_dir output_qwen：指定输出目录，这里是将训练结果保存到名为"output_qwen"的文件夹中。
- --num_train_epochs 5：指定训练的轮数，这里是训练5轮。
- --per_device_train_batch_size 2：指定每个设备（如GPU）上用于训练的批次大小，这里是每个设备上训练2个样本。
- --per_device_eval_batch_size 1：指定每个设备上用于评估的批次大小，这里是每个设备上评估1个样本。
- --gradient_accumulation_steps 8：指定梯度累积步数，这里是梯度累积8步后再更新模型参数。
- --evaluation_strategy "no"：指定评估策略，这里是不进行评估。
- --save_strategy "steps"：指定保存策略，这里是每隔一定步数（如1000步）保存一次模型。
- --save_steps 1000：指定保存步数，这里是每隔1000步保存一次模型。
- --save_total_limit 10：指定最多保存的模型数量，这里是最多保存10个模型。
- --learning_rate 3e-4：指定学习率，这里是3e-4。
- --weight_decay 0.1：指定权重衰减系数，这里是0.1。
- --adam_beta2 0.95：指定Adam优化器的beta2参数，这里是0.95。
- --warmup_ratio 0.01：指定预热比例，这里是预热比例为总步数的1%。
- --lr_scheduler_type "cosine"：指定学习率调度器类型，这里是余弦退火调度器。
- --logging_steps 1：指定日志记录步数，这里是每1步记录一次日志。
- --report_to "none"：指定报告目标，这里是不报告任何信息。
- --model_max_length 512：指定模型的最大输入长度，这里是512个字符。
- --lazy_preprocess True：指定是否使用懒加载预处理，这里设置为True。
- --gradient_checkpointing：启用梯度检查点技术，可以在训练过程中节省显存并加速训练。
- --use_lora：指定是否使用LORA（Layer-wise Relevance Analysis）技术，这里设置为True


```python
%%time
!python ./Qwen/finetune.py \
--model_name_or_path "/mnt/workspace/.cache/modelscope/qwen/Qwen-1_8B-Chat/" \
--data_path qwen_chat.json \
--fp16 True \
--output_dir output_qwen \
--num_train_epochs 10 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 8 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 1000 \
--save_total_limit 10 \
--learning_rate 3e-4 \
--weight_decay 0.1 \
--adam_beta2 0.95 \
--warmup_ratio 0.01 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--report_to "none" \
--model_max_length 512 \
--lazy_preprocess True \
--gradient_checkpointing True \
--use_lora True
```

    [2024-03-16 16:32:02,034] [INFO] [real_accelerator.py:158:get_accelerator] Setting ds_accelerator to cuda (auto detect)
    2024-03-16 16:32:03.298260: I tensorflow/core/util/port.cc:111] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-03-16 16:32:03.328849: E tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:9342] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    2024-03-16 16:32:03.328873: E tensorflow/compiler/xla/stream_executor/cuda/cuda_fft.cc:609] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    2024-03-16 16:32:03.328894: E tensorflow/compiler/xla/stream_executor/cuda/cuda_blas.cc:1518] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    2024-03-16 16:32:03.334113: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-03-16 16:32:04.014023: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    The model is automatically converting to bf16 for faster inference. If you want to disable the automatic precision, please manually add bf16/fp16/fp32=True to "AutoModelForCausalLM.from_pretrained".
    Try importing flash-attention for faster inference...
    Warning: import flash_attn rms_norm fail, please install FlashAttention layer_norm to get higher efficiency https://github.com/Dao-AILab/flash-attention/tree/main/csrc/layer_norm
    Loading checkpoint shards: 100%|██████████████████| 2/2 [00:00<00:00,  2.41it/s]
    trainable params: 53,673,984 || all params: 1,890,502,656 || trainable%: 2.83913824874309
    Loading data...
    Formatting inputs...Skip in lazy mode
    Detected kernel version 4.19.24, which is below the recommended minimum of 5.5.0; this can cause the process to hang. It is recommended to upgrade the kernel to the minimum version or higher.
      0%|                                                    | 0/10 [00:00<?, ?it/s]/opt/conda/lib/python3.10/site-packages/torch/utils/checkpoint.py:429: UserWarning: torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly. The default value of use_reentrant will be updated to be False in the future. To maintain current behavior, pass use_reentrant=True. It is recommended that you use use_reentrant=False. Refer to docs for more details on the differences between the two variants.
      warnings.warn(
    {'loss': 0.1271, 'learning_rate': 0.0003, 'epoch': 1.0}                         
    {'loss': 0.1271, 'learning_rate': 0.0002909538931178862, 'epoch': 2.0}          
    {'loss': 0.04, 'learning_rate': 0.00026490666646784665, 'epoch': 3.0}           
    {'loss': 0.0029, 'learning_rate': 0.000225, 'epoch': 4.0}                       
    {'loss': 0.0005, 'learning_rate': 0.00017604722665003956, 'epoch': 5.0}         
    {'loss': 0.0005, 'learning_rate': 0.00012395277334996044, 'epoch': 6.0}         
    {'loss': 0.0006, 'learning_rate': 7.500000000000002e-05, 'epoch': 7.0}          
    {'loss': 0.0005, 'learning_rate': 3.509333353215331e-05, 'epoch': 8.0}          
    {'loss': 0.0006, 'learning_rate': 9.046106882113751e-06, 'epoch': 9.0}          
    {'loss': 0.0005, 'learning_rate': 0.0, 'epoch': 10.0}                           
    {'train_runtime': 6.2593, 'train_samples_per_second': 4.793, 'train_steps_per_second': 1.598, 'train_loss': 0.030027845277800225, 'epoch': 10.0}
    100%|███████████████████████████████████████████| 10/10 [00:06<00:00,  1.60it/s]
    CPU times: user 110 ms, sys: 36.9 ms, total: 147 ms
    Wall time: 15.4 s


## 6.模型合并

与全参数微调不同，LoRA和Q-LoRA的训练只需存储adapter部分的参数。使用LoRA训练后的模型，可以选择先合并并存储模型（LoRA支持合并，Q-LoRA不支持），再用常规方式读取你的新模型。

```python
%%time
from peft import AutoPeftModelForCausalLM 
from transformers import AutoTokenizer 


# 分词
tokenizer = AutoTokenizer.from_pretrained("output_qwen", trust_remote_code=True ) 
tokenizer.save_pretrained("qwen-1_8b-finetune")

# 模型
model = AutoPeftModelForCausalLM.from_pretrained("output_qwen", device_map="auto", trust_remote_code=True ).eval() 
merged_model = model.merge_and_unload() 
merged_model.save_pretrained("qwen-1_8b-finetune", max_shard_size="2048MB", safe_serialization=True) # 最大分片2g


```

    The model is automatically converting to bf16 for faster inference. If you want to disable the automatic precision, please manually add bf16/fp16/fp32=True to "AutoModelForCausalLM.from_pretrained".
    Try importing flash-attention for faster inference...
    Warning: import flash_attn rms_norm fail, please install FlashAttention layer_norm to get higher efficiency https://github.com/Dao-AILab/flash-attention/tree/main/csrc/layer_norm
    Loading checkpoint shards: 100%|██████████| 2/2 [00:00<00:00,  2.43it/s]


    CPU times: user 10.2 s, sys: 3.06 s, total: 13.2 s
    Wall time: 12.7 s


## 7.本地部署微调模型

使用微调后且合并的模型进行本地部署。

```python
%%time
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig 


query = "识别以下句子中的地址信息，并按照{address:['地址']}的格式返回。如果没有地址，返回{address:[]}。句子为：在一本关于人文的杂志中，我们发现了一篇介绍北京市海淀区科学院南路76号社区服务中心一层的文章，文章深入探讨了该地点的人文历史背景以及其对于当地居民的影响。"
local_model_path = "qwen-1_8b-finetune"
tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(local_model_path, device_map="auto", trust_remote_code=True).eval()
response, history = model.chat(tokenizer, query, history=None)
print("回答如下:\n", response)
```

    Warning: import flash_attn rms_norm fail, please install FlashAttention layer_norm to get higher efficiency https://github.com/Dao-AILab/flash-attention/tree/main/csrc/layer_norm
    Loading checkpoint shards: 100%|██████████| 2/2 [00:00<00:00,  2.03it/s]


    回答如下:
     {"address":"北京市海淀区科学院南路76号社区服务中心一层"}
    CPU times: user 1.66 s, sys: 269 ms, total: 1.93 s
    Wall time: 1.93 s


## 8.保存依赖包信息


```python
!pip freeze > requirements_qwen_1_8.txt
```

## 参考资料

- https://www.modelscope.cn/models/qwen/Qwen-1_8B-Chat/summary
- https://gitcode.com/QwenLM/Qwen.git
- https://blog.csdn.net/qq_45156060/article/details/135153920
- https://blog.csdn.net/weixin_44750512/article/details/135099562


```python

```


```python

```
