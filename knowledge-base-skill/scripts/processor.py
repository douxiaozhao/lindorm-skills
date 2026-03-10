import argparse
import sys
from tqdm import tqdm
from main_logic import Lindorm, AliQwen
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

def init_env(index_name):
    """初始化环境"""
    ld = Lindorm(index_name)
    ld.lindormAI.create_embedding_model()
    ld.lindormAI.create_reranker_model()
    ld.lindormSearch.create_pipelines()
    ld.lindormSearch.create_indices()
    print("Lindorm Search-based environment initialized.")

def ingest_txt(index_name, file_path):
    """处理 TXT 并入库（仅限搜索索引）"""
    ld = Lindorm(index_name)
    
    print(f"📚 加载文档: {file_path}")
    loader = TextLoader(file_path, autodetect_encoding=True)
    
    # 1. 父级切分（1000字）
    parent_split = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = loader.load_and_split(parent_split)
    
    print(f"✂️  父级切分: {len(docs)} 个文档块")
    print(f"📦 开始导入...\n")
    
    # 2. 子级切分（250字）
    child_split = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=0)
    
    total_chunks = 0
    for i, doc in enumerate(tqdm(docs, desc="处理父文档", unit="doc")):
        p_id = f"{file_path}_p_{i}"
        # 写入父索引
        ld.lindormSearch.write_parent({"document_id": p_id, "context": doc.page_content})
        
        # 写入子索引
        chunks = child_split.split_text(doc.page_content)
        write_chunks = [{"document_id": p_id, "text_field": c, "chunking_position": j} for j, c in enumerate(chunks)]
        ld.lindormSearch.write_chunking_bulk(write_chunks)
        total_chunks += len(chunks)
    
    print(f"\n✅ 导入完成！")
    print(f"   - 父文档: {len(docs)} 个")
    print(f"   - 子切片: {total_chunks} 个")

def chat_rag(index_name, query):
    """Rerank + Parent Context RAG"""
    ld = Lindorm(index_name)
    qwen = AliQwen()
    
    # 1. 混合检索
    res = ld.lindormSearch.rrf_search(query, k=10)
    # 2. Rerank
    texts = [item["_source"]["text_field"] for item in res]
    rerank_res = ld.lindormAI.reranker(query, texts)
    # 3. 关联父表全文
    top_hit = rerank_res[0]
    p_id = res[top_hit['index']]['_source']['document_id']
    full_context = ld.lindormSearch.get_parent_context(p_id)
    
    # 4. 回答
    prompt = qwen.PROMPT_TEMPLATE.format(context=full_context, question=query)
    for part in qwen.chat_stream(prompt):
        sys.stdout.write(part)
        sys.stdout.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["init", "ingest", "chat"])
    parser.add_argument("--index", default="test_kb")
    parser.add_argument("--file_path")
    parser.add_argument("--query")
    args = parser.parse_args()

    if args.command == "init": init_env(args.index)
    elif args.command == "ingest": ingest_txt(args.index, args.file_path)
    elif args.command == "chat": chat_rag(args.index, args.query)