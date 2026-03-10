---
name: lindorm-knowledge-base
description: 基于 Lindorm 的 RAG 知识库管理技能。支持中文文档智能切分、父子表级联存储、RRF 混合检索以及基于通义千问的增强检索问答。
---

# Lindorm Knowledge Base Skill

## 🎯 技能能力
本技能实现了完整的 RAG (检索增强生成) 工作流：
1. **环境初始化**：自动创建 Lindorm 父子表、部署 BGE 向量模型与 Rerank 模型。
2. **文档处理**：支持 `ChineseTextSplitter`（按中文标点切分）或 `Recursive`（按长度切分）进行文档分片。
3. **数据入库**：父表存储原始文档，子表存储切片及其向量，并自动同步至搜索引擎。
4. **混合检索问答**：支持 RRF 融合检索 + Rerank 重排 + 通义千问流式回答。

# 🚀 执行指令规范

## 🛠️ 环境依赖
在运行任何指令前，Agent 需确保当前环境已安装以下 Python 库
进入工作目录
```bash
cd ~/.claude/skills/lindorm-knowledge-base
source venv/bin/activate
```

### 1. 初始化知识库环境
当用户要求"准备知识库环境"或"初始化 Lindorm 表"时：
```bash
python scripts/processor.py init
```

### 2. 导入文档并自动切片
当用户提供 JSON 格式文档（需包含 id, title, context）并要求入库时：
```bash
python scripts/processor.py ingest --file_path <json_path>
```

### 3. 知识库 RAG 问答
当用户提出问题并要求根据知识库回答时：
```bash
# --mode 指定检索粒度：child (切片问答) 或 parent (关联全文问答)
python scripts/processor.py chat --query <问题>
```