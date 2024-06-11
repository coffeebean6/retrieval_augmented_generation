"""
应该中所用到的模型操作
"""
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from FlagEmbedding import FlagReranker
from langchain_community.chat_models import ChatOllama
from typing import List

"""
句子embedding模型
"""
class SentenceEmbeddingModel():
    MODEL_NAME = "moka-ai/m3e-base"
    embeddings: HuggingFaceBgeEmbeddings
    
    def __init__(self) -> None:
        print(f"初始化SentenceEmbedding模型：{self.MODEL_NAME}")
        # embedding model
        model_name  = self.MODEL_NAME
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            # model_kwargs=model_kwargs,
            # encode_kwargs=encode_kwargs
        )
        self.embeddings = embeddings
    
    """embedding 单句"""
    def embed_query(self, query:str) -> List[float]:
        key_feat = self.embeddings.embed_query(query)
        return key_feat
    
    """embedding 一批句子"""
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        list_feat = self.embeddings.embed_documents(texts)
        return list_feat


"""
embedding重排序模型
"""
class RerankModel():
    MODEL_NAME = 'BAAI/bge-reranker-base'
    reranker: FlagReranker
    
    def __init__(self) -> None:
        print(f"初始化Rerank模型：{self.MODEL_NAME}")
        model_name  = self.MODEL_NAME
        reranker = FlagReranker(model_name)
        self.reranker = reranker
    
    """query跟一批句子做比较，返回相似度最高的 top_k 条""" 
    def rank(self, query:str, texts:List[str], top_k=3) -> List[str]:
        pairs = [[query, text] for text in texts]
        scores = self.reranker.compute_score(pairs)
        combined = list(zip(scores, pairs))
        sorted_combined = sorted(combined, reverse=True)
        sorted_pairs = [item[1] for item in sorted_combined]
        return sorted_pairs[:top_k]


"""
对话模型
"""
class ChatModel():
    MODEL_NAME = 'wangshenzhi/llama3-8b-chinese-chat-ollama-q4:latest'
    llm: ChatOllama
    
    def __init__(self) -> None:
        self.llm = ChatOllama(
            base_url="http://localhost:11434", 
            model=self.MODEL_NAME
        )
        print(f"初始化chat模型：{self.MODEL_NAME}")
        # 模型初始化
        self.llm.invoke("hello")
    
    def get_model(self) -> ChatOllama:
        return self.llm
    
    