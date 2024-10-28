# aiki

[![codecov](https://codecov.io/gh/OpenKILab/AIKI/branch/main/graph/badge.svg?token=AIKI_token_here)](https://codecov.io/gh/OpenKILab/AIKI)
[![CI](https://github.com/OpenKILab/AIKI/actions/workflows/main.yml/badge.svg)](https://github.com/OpenKILab/AIKI/actions/workflows/main.yml)

Awesome aiki created by OpenKILab

## Install it from PyPI

```bash
pip install aiki
```

## Usage

```py
from aiki import KnowledgeBase, Blob

kb = KnowledgeBase(name = "Environment", retriever: "BM25 Retrever", reranker="Cross-Encoder", judger="SKR Judger", refiner="Selective-Context Refiner", delete_algo: Callable= similarity_func)

# 1. add
# 2. update
# 3. delete
# 4. query
# 5. config

# 1. add
kb.add(data="xxx")
# Create knowledge: Blob -> Knowledge

# 2. update
kb.update(data="xxx")

# 3. delete
kb.delete(data="xxx")

# 4. query
result = kb.query(query="what's the weather like in shanghai", 
    args:Optional[Dict]= {top_k: 5, modal_type: ["text", "image"]})
# result: {"text": ["text result 0", ...], "image" : ["base64 encoded img 0", ...]}

```

## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.
