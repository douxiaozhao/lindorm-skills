# -*- coding: utf-8 -*-
import argparse
import csv
import io
import json
import base64
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入您原有的模块
from lindorm import Lindorm
from index import get_index_body
from prompt import VL_PROMPT, REWRITE_SUMMARY_PROMPT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("opensearch").setLevel(logging.ERROR)

# 默认模型配置 (参考原 Notebook)
VL_MODEL = 'qwen3-vl-plus'
EMBEDDING_MODEL = 'qwen2.5-vl-embedding'
REWRITE_MODEL = 'qwen-plus'
RERANK_MODEL = 'qwen3-rerank'

def init_index(index_name: str):
    """初始化/创建索引"""
    lindorm = Lindorm(index_name)
    res = lindorm.lindormSearch.get_index()
    if res:
        logging.info(f"Index {index_name} already exists.")
        return
    
    index_body = get_index_body(1024)
    result = lindorm.lindormSearch.create_search_index(index_body)
    logging.info(f"Index created: {result}")

def safe_json_loads(s):
    try:
        # 去除 markdown 代码块包裹
        s = s.strip()
        if s.startswith("```") and s.endswith("```"):
            s = s.split("\n", 1)[1].rsplit("\n", 1)[0]
        return json.loads(s)
    except Exception as e:
        logging.error(f"JSON Parsing failed: {e}")
        return {"content": s}

def process_row_safe(row, lindorm: Lindorm):
    """处理单行数据入库"""
    doc_id = row.get('id')
    row_copy = dict(row)
    del row_copy['id']

    if lindorm.lindormSearch.get_doc(doc_id):
        return f"Skip: {doc_id} (exists)"

    try:
        # 1. 获取向量
        embedding = lindorm.lindormAI.embedding("image", row_copy['url'], EMBEDDING_MODEL)
        # 2. VL 模型生成描述
        vl_desc = lindorm.lindormAI.vl_picture_withdraw(row_copy['url'], VL_MODEL, VL_PROMPT)
        vl_desc_json = safe_json_loads(vl_desc)
        # 3. 改写描述
        vl_desc_rewrite = lindorm.lindormAI.rewrite_text(vl_desc_json.get('content', ''), REWRITE_MODEL, REWRITE_SUMMARY_PROMPT)
        
        row_copy['embedding'] = embedding
        row_copy['img_desc'] = vl_desc_rewrite
        
        # 4. 写入 Lindorm
        lindorm.lindormSearch.write_doc(row_copy, doc_id)
        return f"Success: {doc_id}"
    except Exception as e:
        raise Exception(f"Error processing row {doc_id}: {e}")

def ingest_csv(index_name: str, csv_path: str):
    """CSV 文件入库"""
    lindorm = Lindorm(index_name)
    logging.info(f"Processing CSV: {csv_path}")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        csv_reader = list(csv.DictReader(f))
        
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_row = {executor.submit(process_row_safe, row, lindorm): row for row in csv_reader}
        for future in as_completed(future_to_row):
            try:
                result = future.result()
                logging.info(result)
            except Exception as e:
                logging.error(f"Row failed: {e}")
                
    # 触发构建离线索引
    lindorm.lindormSearch.build_index("embedding")
    logging.info("Ingestion complete and offline index build triggered.")

def search_text(index_name: str, query: str, top_k: int):
    """文本搜图 (RRF 融合检索 + Rerank)"""
    lindorm = Lindorm(index_name)
    # 获取文本向量
    text_embedding = lindorm.lindormAI.embedding("text", query, EMBEDDING_MODEL)
    
    # RRF 融合检索 (依赖 text_embedding 和 img_desc)
    hits = lindorm.lindormSearch.rrf_search(query, "img_desc", text_embedding, "embedding", True, factor=0.3, min_score=0.5, top_k=top_k * 2)
    
    if not hits:
        print("未找到匹配的图片。")
        return

    # 重排 (Rerank)
    description_chunks = [hit.get('_source').get("img_desc", "") for hit in hits]
    rerank_result = lindorm.lindormAI.rerank_text(query, description_chunks, RERANK_MODEL, top_k)
    
    print("\n=== 多模态检索结果 (基于 RRF + Rerank) ===")
    for rerank_hit in rerank_result:
        hit = hits[rerank_hit.get('index')]
        source = hit.get('_source', {})
        print(f"ID: {hit.get('_id')}")
        print(f"Score: {rerank_hit.get('relevance_score'):.4f}")
        print(f"URL: {source.get('url')}")
        print(f"Description: {source.get('img_desc')[:50]}...\n")

def search_image(index_name: str, image_path: str, top_k: int):
    """以图搜图 (纯 KNN 检索)"""
    lindorm = Lindorm(index_name)
    
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    image_data = f"data:image/jpeg;base64,{encoded_string}"
    
    # 获取图片向量
    image_embedding = lindorm.lindormAI.embedding("image", image_data, EMBEDDING_MODEL)
    
    # 纯 KNN 向量检索
    hits = lindorm.lindormSearch.knn_search(image_embedding, "embedding", True, min_score=0.5, top_k=top_k)
    
    print("\n=== 以图搜图结果 (基于 KNN) ===")
    for hit in hits:
        source = hit.get('_source', {})
        print(f"ID: {hit.get('_id')}, Score: {hit.get('_score'):.4f}, URL: {source.get('url')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lindorm Multimodal Search Skill")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # 共用参数
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--index", type=str, default="multimodal_retrieval_index", help="Index name")
    
    # Init 
    parser_init = subparsers.add_parser("init", parents=[common_parser])
    
    # Ingest
    parser_ingest = subparsers.add_parser("ingest", parents=[common_parser])
    parser_ingest.add_argument("--csv_path", type=str, required=True, help="Path to CSV data")
    
    # Search Text
    parser_search_text = subparsers.add_parser("search_text", parents=[common_parser])
    parser_search_text.add_argument("--query", type=str, required=True, help="Text query")
    parser_search_text.add_argument("--top_k", type=int, default=5, help="Number of results")
    
    # Search Image
    parser_search_img = subparsers.add_parser("search_image", parents=[common_parser])
    parser_search_img.add_argument("--image_path", type=str, required=True, help="Path to local image file")
    parser_search_img.add_argument("--top_k", type=int, default=5, help="Number of results")
    
    args = parser.parse_args()
    
    if args.command == "init":
        init_index(args.index)
    elif args.command == "ingest":
        ingest_csv(args.index, args.csv_path)
    elif args.command == "search_text":
        search_text(args.index, args.query, args.top_k)
    elif args.command == "search_image":
        search_image(args.index, args.image_path, args.top_k)