# -*- coding: utf-8 -*-
import json
import requests
from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk
from typing import List
from http import HTTPStatus
from dashscope import Generation
from ldconfig import Config

class Lindorm:
    def __init__(self, index_name):
        self.index_name = index_name
        self.embedding_model_name = "bge_m3_model"    
        self.reranker_model_name = "rerank_bge_v2_m3"
        self.lindormAI = self.LindormAI(self)
        self.lindormSearch = self.LindormSearch(self)

    class LindormAI:
        """AI 引擎 Rest 接口封装"""
        def __init__(self, parent):
            self.parent = parent
            self.headers = {
                "Content-Type": "application/json; charset=utf-8",
                "x-ld-ak": Config.LD_USER,
                "x-ld-sk": Config.LD_PASSWORD
            }
            
        def list_modes(self) -> list:
            url = f"http://{Config.AI_HOST}:{Config.AI_PORT}/v1/ai/models/list"
            response = requests.get(url, headers=self.headers)
            return response.json()["data"]["models"]
        
        def create_embedding_model(self):
            url = f"http://{Config.AI_HOST}:{Config.AI_PORT}/v1/ai/models/create"
            data = {"model_name": self.parent.embedding_model_name, "model_path": "huggingface://BAAI/bge-m3", "task": "FEATURE_EXTRACTION", "algorithm": "BGE_M3", "settings": {"instance_count": "2"}}
            requests.post(url, json=data, headers=self.headers)
          
        def create_reranker_model(self):
            url = f"http://{Config.AI_HOST}:{Config.AI_PORT}/v1/ai/models/create"
            data = {"model_name": self.parent.reranker_model_name, "model_path": "huggingface://BAAI/bge-reranker-v2-m3", "task": "SEMANTIC_SIMILARITY", "algorithm": "BGE_RERANKER_V2_M3"}
            requests.post(url, json=data, headers=self.headers)

        def reranker(self, query: str, chunks: List[str]):
            url = f"http://{Config.AI_HOST}:{Config.AI_PORT}/v1/ai/models/{self.parent.reranker_model_name}/infer"
            return requests.post(url, json={"input": {"query": query, "chunks": chunks}}, headers=self.headers).json()["data"]
        
        def handler_reranker(self, origin_result, reranker_result, topk):
            reranked = []
            for item in reranker_result:
                idx = item['index']
                if idx < len(origin_result):
                    original = origin_result[idx]
                    original['rerank_score'] = item['score']
                    reranked.append(original)
            return reranked[0:topk]

    class LindormSearch:
        """搜索引擎操作封装（基于 OpenSearch 客户端）"""
        def __init__(self, parent):
            self.parent = parent
            self.parent_index = self.parent.index_name + "_parent"
            self.chunking_index = self.parent.index_name + "_chunking"
            self.text_field = "text_field" 
            self.vector_field = "vector_field" 
            self.write_pipeline = "demo_write_embedding_pipeline" 
            self.search_pipeline = "demo_search_embedding_pipeline"
            
            self.client = OpenSearch(
                hosts=[{"host": Config.SEARCH_HOST, "port": Config.SEARCH_PORT}],
                http_auth=(Config.LD_USER, Config.LD_PASSWORD),
                use_ssl=False, timeout=60
            )

        def create_pipelines(self):
            """创建写入与搜索 Pipeline"""
            inner_ai = Config.AI_HOST.replace("-pub", "-vpc")
            url = f"http://{inner_ai}:{int(Config.AI_PORT)}"
            
            # 写入 Pipeline
            write_p = {"description": "ingest pipeline", "processors": [{"text-embedding": {"inputFields": [self.text_field], "outputFields": [self.vector_field], "userName": Config.LD_USER, "password": Config.LD_PASSWORD, "url": url, "modeName": self.parent.embedding_model_name}}]}
            self.client.ingest.put_pipeline(id=self.write_pipeline, body=write_p)
            
            # 搜索 Pipeline
            search_p = {"request_processors": [{"text-embedding": {"tag": "auto-query-embedding", "model_config": {"inputFields": [self.text_field], "outputFields": [self.vector_field], "userName": Config.LD_USER, "password": Config.LD_PASSWORD, "url": url, "modeName": self.parent.embedding_model_name}}}]}
            self.client.search_pipeline.put(id=self.search_pipeline, body=search_p)

        def create_indices(self):
            """创建父子表索引"""
            # 父表索引
            if not self.client.indices.exists(self.parent_index):
                self.client.indices.create(index=self.parent_index, body={"mappings": {"properties": {"document_id": {"type": "keyword"}, "context": {"type": "text", "index": False}}}})
            
            # 子表索引（带 KNN 向量）
            if not self.client.indices.exists(self.chunking_index):
                body = {
                    "settings": {"index": {"knn": True, "default_pipeline": self.write_pipeline, "search.default_pipeline": self.search_pipeline}},
                    "mappings": {"_source": {"excludes": [self.vector_field]}, "properties": {self.vector_field: {"type": "knn_vector", "dimension": 1024, "method": {"engine": "lvector", "name": "hnsw", "space_type": "l2", "parameters": {"m": 24, "ef_construction": 500}}}} }
                }
                self.client.indices.create(index=self.chunking_index, body=body)

        def write_parent(self, data):
            self.client.index(index=self.parent_index, id=data['document_id'], body=data)
            
        def write_chunking_bulk(self, datas):
            """批量写入切片"""
            def gen():
                for data in datas:
                    yield {"_op_type": "index", "_index": self.chunking_index, "_id": f"{data['document_id']}_{data['chunking_position']}", "document_id": data['document_id'], "text_field": data['text_field']}
            bulk(self.client, gen(), chunk_size=500)

        def rrf_search(self, text_query, k=5):
            """融合检索"""
            body = {
                "size": k, "_source": ["document_id", "text_field"],
                "query": {"knn": {self.vector_field: {"query_text": text_query, "filter": {"match": {self.text_field: text_query}}, "k": k}}},
                "ext": {"lvector": {"hybrid_search_type": "filter_rrf", "rrf_rank_constant": "1"}}
            }
            return self.client.search(index=self.chunking_index, body=body)['hits']['hits']

        def get_parent_context(self, doc_id):
            return self.client.get(index=self.parent_index, id=doc_id)['_source']['context']

class AliQwen:
    """通义千问对接"""
    def __init__(self):
        self.api_key = Config.DASHSCOPE_API_KEY
        self.PROMPT_TEMPLATE = "已知信息：\n{context}\n回答问题：{question}"
    
    def chat_stream(self, prompt: str):
        responses = Generation.call(model="qwen-plus", prompt=prompt, stream=True, api_key=self.api_key)
        for response in responses:
            if response.status_code == HTTPStatus.OK:
                yield response.output.text