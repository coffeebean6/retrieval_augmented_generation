# Retrieval Augmented Generation
A simple Retrieval Augmented Generation function uses M3e model to extract the features of documents, and uses Milvus as a vector database to store and query similar images.  

<br/>
  
You can run the following script to start the Gradio web UI:
```bash
python3 rag.py
```
  
中文解释文档在<a href="document_zh.pdf">这里</a>。
<br/>

---

<br/>

If you can't run it directly, you may need to do some preparation, including but not limilited to:

- Install Ollama and run LLM:
```bash
https://ollama.com/ download and install.

> ollama run wangshenzhi/llama3-8b-chinese-chat-ollama-q4
```


- Install or update vector database:
```bash
$ wget https://github.com/milvus-io/milvus/releases/download/v2.4.4/milvus-standalone-docker-compose.yml -O docker-compose.yml
$ docker-compose up -d
$ docker ps
```

- Install package:
```bash
pip install -U gradio pymilvus transformers FlagEmbedding langchain langchain-core langchain_community langchain-milvus langchain-text-splitters pypdf2 bs4 
```
