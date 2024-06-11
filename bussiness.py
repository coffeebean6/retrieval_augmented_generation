"""
业务处理相关  
"""
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Any
from vectordbs import MilvusDB
from models import SentenceEmbeddingModel, RerankModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_community.chat_models import ChatOllama
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate


"""
文档处理
"""
class DocumentHandler():
    VDB_HOST = '127.0.0.1'
    VDB_PORT = 19530
    COLLECTION_NAME = 'liam_text_06'
    DIM = 768
    
    vectordb: MilvusDB
    embeddingModel: SentenceEmbeddingModel
    
    def __init__(self, vectordb:MilvusDB, embeddingModel:SentenceEmbeddingModel) -> None:
        self.vectordb = vectordb
        self.embeddingModel = embeddingModel
    
    """从web端上传文件"""
    def upload_file(self, read_file_path:str, write_file_path:str) -> str:
        with open(read_file_path, 'rb') as r_file:
            content = r_file.read()
            with open(write_file_path, 'wb') as w_file:
                w_file.write(content)
        return write_file_path
    
    """装载和切分文档"""
    def load_and_split(self, file_path:str) -> List[Document]:
        doc_list : List[Document]
        if file_path.lower().endswith('.pdf'):
            doc_list = PyPDFLoader(file_path).load()
        else:
            raise ValueError("This type of file can NOT be supported.")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap=20)
        documents = text_splitter.split_documents(doc_list)
        return documents
    
    """embedding并存储到向量数据库中"""
    def embed_and_store_vector(self, documents:List[Document]) -> int:
        # 建表和索引
        self.vectordb.create_collection(self.COLLECTION_NAME, self.DIM)
        # 生成向量
        source = [d.metadata['source'] for d in documents]
        page = [d.metadata['page'] for d in documents]
        texts = [d.page_content for d in documents]
        vectors = self.embeddingModel.embed_documents(texts=texts)
        # 存入句子和向量
        insert_list = [source, page, texts, vectors]
        nums = self.vectordb.insert_data(insert_list, self.COLLECTION_NAME)
        return nums
    
    """获取支持的doc列表"""
    def get_doc_file_list(self) -> List[str]:
        return self.vectordb.search_source(self.COLLECTION_NAME, self.DIM)


"""
定义自己的 retriever 类，以便于灵活处理rerank
"""
class MyMilvusRerankRetriever(BaseRetriever):
    vectordb: Any
    embeddingModel: Any
    rerankModel: Any
    top_k: int = 20
    top_n: int = 3
    
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
    
    # 设置模型相关
    def set(self, vectordb:MilvusDB, embeddingModel:SentenceEmbeddingModel, rerankModel:RerankModel) -> None:
        self.vectordb = vectordb
        self.embeddingModel = embeddingModel
        self.rerankModel = rerankModel
    
    # 设置从db取的相似条数
    def set_search_result_num(self,top_k:int) -> None:
        self.top_k = top_k
    
    # 设置rerank后取的条数
    def set_rerank_result_num(self,top_n:int) -> None:
        self.top_n = top_n
        
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # 对query进行embedding
        if self.embeddingModel is None:
            raise ValueError("embedding model error")
        key_feature = self.embeddingModel.embed_query(query)
        # 从db中获取初步结果
        if self.vectordb is None:
            raise ValueError("vectordb error")
        text_search_result:List[str] = self.vectordb.search_data(DocumentHandler.COLLECTION_NAME, key_feature, topk=self.top_k)
        print("---------Embedding检索出的结果>>>>", text_search_result)
        if len(text_search_result) == 0:
            return []
        # rerank重排序
        if self.rerankModel is None:
            raise ValueError("rerank model error")
        rerank_result: List[str] = self.rerankModel.rank(query, text_search_result)
        print("=========Rerank出的最终结果>>>>", rerank_result)
        # 构造成 Document 返回
        if rerank_result:
            return [Document(page_content=doc[1]) for doc in rerank_result]
        return []
    
    
class Chat():
    QA_PROMPT="""使用以下上下文和聊天历史记录来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案.
    上下文: {context}
    
    聊天历史记录: {chat_history}
    
    问题: {question}
    有帮助的答案:"""
    prompt = PromptTemplate(
        template=QA_PROMPT, input_variables=["context", "chat_history", "question"]
    )
    llm: ChatOllama
    
    def __init__(self, llm: ChatOllama) -> None:
        self.llm = llm
    
    def chat_with_history(self, message:str, history, retriever:MyMilvusRerankRetriever) -> str:
        chain = load_qa_chain(self.llm, chain_type="stuff", prompt=self.prompt)
        context = retriever.get_relevant_documents(message)
        result = chain(
            {
                "input_documents": context, 
                "question": message, 
                "chat_history": history
            }, 
            return_only_outputs=False
        )
        return result['output_text']