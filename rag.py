"""
UI界面
"""
import gradio as gr
import os
from bussiness import DocumentHandler, MyMilvusRerankRetriever, Chat
from vectordbs import MilvusDB
from models import SentenceEmbeddingModel, RerankModel, ChatModel

# 初始化模型
embeddingModel: SentenceEmbeddingModel = SentenceEmbeddingModel()
rerankModel: RerankModel = RerankModel()
chatModel: ChatModel = ChatModel()
vectordb: MilvusDB = MilvusDB()
# 初始化业务逻辑
document_handler = DocumentHandler(vectordb=vectordb, embeddingModel=embeddingModel)
myRetriever: MyMilvusRerankRetriever = MyMilvusRerankRetriever()
myRetriever.set(vectordb, embeddingModel, rerankModel)
myRetriever.set_search_result_num(top_k=20)
myRetriever.set_rerank_result_num(top_n=3)
chat: Chat = Chat(chatModel.get_model())

# 获取rag支持的文件列表
file_list = document_handler.get_doc_file_list()

# 确保uploads目录存在
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# 处理文件上传
def upload_file(file):
    read_file_path = file.name
    doc_name = os.path.basename(read_file_path)
    # 上传文件
    written_file_path = document_handler.upload_file(read_file_path, os.path.join('uploads', doc_name))
    # 装载并切分
    list_doc = document_handler.load_and_split(file_path=written_file_path)
    # 提取特征embedding并存入向量数据库
    vector_num = document_handler.embed_and_store_vector(list_doc)
    global file_list
    # 追加到界面下拉框中
    file_list.append(doc_name)
    return f"{doc_name} uploaded successfully. ({vector_num})", gr.Dropdown(choices=file_list, interactive=True, value=file_list[-1])

# 处理search和chat
def chat_response(message, history):
    res = chat.chat_with_history(message, history, myRetriever)
    return res

# Gradio界面
with gr.Blocks() as demo:
    dropdown = gr.Dropdown(
        choices=file_list, label="Knowledge Base", info="Supported documents: ", interactive=True
    )
    file_upload = gr.File(label="Add a document", type="filepath", height=10)
    upload_button = gr.Button("Upload")
    upload_button.click(upload_file, inputs=file_upload, outputs=[gr.Textbox(label="Upload Status"), dropdown])

    # 聊天框
    chat_interface = gr.ChatInterface(fn=chat_response, title="Chat Interface", fill_height=False, examples=[
        "汕头市东海岸投资建设有限公司负责人叫什么?",
        "塘桥街道社区文化活动中心在哪里?",
        "What is the person in charge of Shantou East Coast Investment and Construction Co., LTD.?"
    ])

if __name__ == '__main__':
    demo.launch()