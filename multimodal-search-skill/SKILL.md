---
name: lindorm-multimodal-search
description: 基于阿里云 Lindorm (OpenSearch) 和通义千问大模型的多模态检索技能。支持创建向量索引、自动化 CSV 图片入库（含描述生成与向量化）、以及文本搜图和以图搜图功能。
---

# Lindorm 多模态检索技能 (Lindorm Multimodal Search)

## 🎯 技能概述
本技能将复杂的 Lindorm 数据库操作与通义千问（Qwen）多模态大模型链路封装为标准指令。Agent 可以利用此技能完成从底层索引构建到上层高级检索（RRF 融合检索 + Rerank 重排）的全过程。

## 🛠️ 环境依赖
在运行任何指令前，Agent 需确保当前环境已安装以下 Python 库
进入工作目录
```bash
cd ~/.claude/skills/lindorm-multimodal-search
source venv/bin/activate
```

## 🚀 核心指令规范

### 1. 初始化索引 (Initialize Index)
当用户要求“创建表”、“初始化数据库”或“准备搜索环境”时使用。
- **参数说明**: `--index` 为索引名称，默认为 `multimodal_retrieval_index`。
```bash
source venv/bin/activate 
python scripts/processor.py init --index <index_name>
```

### 2. CSV 图片数据入库 (Data Ingestion)
当用户要求“上传 CSV”、“导入图片数据”或“解析图片列表”时使用。
* 基本流程:
    1. 逐行读取 CSV 文件（需包含 id, url, create_time 列）。
    2. 调用 qwen2.5-vl-embedding 获取图片向量。
    3. 调用 qwen3-vl-plus 生成图片详细描述。
    4. 调用 qwen-plus 进行描述文本改写（去除反向描述）。
    5. 数据持久化并触发 Lindorm 离线索引构建。
```bash
python scripts/processor.py ingest --index <index_name> --csv_path <csv_file_path>
```

### 3. 以文搜图 (Text-to-Image Search)
用户输入自然语言描述（如“找一张戴帽子的男人在跑步”）进行检索。
检索逻辑: 执行 RRF (Reciprocal Rank Fusion) 融合检索（语义向量 + 文本关键词），并自动通过 qwen3-rerank 模型进行结果重排。
```bash
python scripts/processor.py search_text --index <index_name> --query "<search_text>" --top_k 5
```

### 4. 以图搜图 (Image-to-Image Search)
用户提供本地图片路径，寻找库中视觉相似的图片。
检索逻辑: 调用多模态 Embedding 接口将图片转化为向量，执行 KNN 近邻检索。
```bash
python scripts/processor.py search_image --index <index_name> --image_path "<local_image_path>" --top_k 5
```