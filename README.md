# Lindorm Skills

基于阿里云 [Lindorm](https://www.aliyun.com/product/apsaradb/lindorm) 的 Agent Skill 集合，提供开箱即用的知识库检索与多模态搜索能力，可直接集成到 AI Agent（如 Qoder）中使用。

## 技能列表

| 技能 | 目录 | 简介 |
|------|------|------|
| Lindorm 知识库检索 | `knowledge-base-skill/` | 基于 RAG 的中文知识库管理，支持文档入库、混合检索与大模型问答 |
| Lindorm 多模态搜索 | `multimodal-search-skill/` | 基于图文向量的多模态检索，支持以文搜图、以图搜图 |

---

## knowledge-base-skill — 知识库检索

### 功能介绍

完整的 RAG（检索增强生成）工作流：

- **环境初始化**：自动创建 Lindorm 父子表，部署 BGE 向量模型与 Rerank 模型
- **文档处理**：支持 `ChineseTextSplitter`（按中文标点切分）或 `Recursive`（按长度切分）进行文档分片
- **数据入库**：父表存储原始文档，子表存储切片及其向量，自动同步至搜索引擎
- **混合检索问答**：RRF 融合检索 + Rerank 重排 + 通义千问流式回答

### 环境准备

**1. 配置环境变量**

复制示例配置并填写真实参数：

```bash
cd knowledge-base-skill
cp env.example env
```

编辑 `env` 文件，填写以下参数：

| 变量 | 说明 |
|------|------|
| `AI_HOST` / `AI_PORT` | Lindorm AI 引擎连接地址 |
| `SEARCH_HOST` / `SEARCH_PORT` | Lindorm Search 引擎连接地址 |
| `LD_USER` / `LD_PASSWORD` | Lindorm 用户名与密码 |
| `DASHSCOPE_API_KEY` | 阿里云百炼 API Key |
| `LOAD_FILE_PATH` | 待导入的 JSON 文档路径 |
| `SEARCH_TOP_K` | 检索返回条数 |
| `LD_VL_MODEL` | 视觉向量模型名称 |
| `LD_TEXT_MODEL` | 文本向量模型名称 |

**2. 安装依赖**

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 使用说明

**初始化知识库环境**

```bash
python scripts/processor.py init
```

**导入文档（JSON 格式，需包含 `id`、`title`、`context` 字段）**

```bash
python scripts/processor.py ingest --file_path data/cmrc2018_train.json
```

**RAG 问答**

```bash
python scripts/processor.py chat --query "你的问题"
```

---

## multimodal-search-skill — 多模态搜索

### 功能介绍

基于 Lindorm 向量引擎与通义千问多模态大模型的图文检索技能：

- **索引构建**：创建支持向量检索的多模态索引
- **图片入库**：读取 CSV 文件，自动生成图片描述与向量并持久化
- **以文搜图**：输入自然语言描述，通过 RRF 融合检索 + Rerank 重排返回匹配图片
- **以图搜图**：输入本地图片路径，通过多模态 Embedding 进行 KNN 近邻检索

### 环境准备

**1. 配置环境变量**

```bash
cd multimodal-search-skill
cp env.example env
```

编辑 `env` 文件，填写以下参数：

| 变量 | 说明 |
|------|------|
| `SEARCH_LINK` | Lindorm Search 连接地址（`host:port`） |
| `AI_LINK` | Lindorm AI 连接地址（`host:port`） |
| `LD_USER` / `LD_PASSWORD` | Lindorm 用户名与密码 |
| `SEARCH_TOP_K` | 检索返回条数 |

**2. 安装依赖**

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 使用说明

**初始化索引**

```bash
python scripts/processor.py init --index <index_name>
```

**导入图片数据（CSV 需包含 `id`、`url`、`create_time` 列）**

```bash
python scripts/processor.py ingest --index <index_name> --csv_path <csv_file_path>
```

**以文搜图**

```bash
python scripts/processor.py search_text --index <index_name> --query "戴帽子的男人在跑步" --top_k 5
```

**以图搜图**

```bash
python scripts/processor.py search_image --index <index_name> --image_path "/path/to/image.jpg" --top_k 5
```

---

## 在 Agent 中部署 Skill

以 Qoder Agent 为例，将 skill 安装到 Agent 环境：

```bash
# 知识库检索 skill
cp -r knowledge-base-skill ~/.qoder/skills/lindorm-knowledge-base

# 多模态搜索 skill
cp -r multimodal-search-skill ~/.qoder/skills/lindorm-multimodal-search
```

安装完成后，在 Agent 对话中即可直接调用对应技能，例如：
- "帮我初始化知识库环境"
- "将 data/cmrc2018_train.json 导入知识库"
- "根据知识库回答：什么是 RAG？"
- "搜索一张猫咪在草地上玩耍的图片"

## 许可证

本项目基于 Apache 2.0 协议开源，详见各 skill 目录下的 `LICENSE.txt`。
