# Retrieval Augmented Generation
A naive Retrieval Augmented Generation implementation uses the M3e model for text embedding, uses Milvus for vector data storage and retrieval, and LLM uses Llama3-8b. 
<br/>
The UI is shown below:

<p align="center">
  <img src="data/rag.png" alt="UI demo" />
</p>

<br/>
  
You can run the following script to start the Gradio web UI:
```bash
python3 rag.py
```

<br/>
A brief description video is below:
<br/>
https://www.bilibili.com/video/BV1ff421X7YN
<br/>
<br/>
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
