from langchain_community.vectorstores import FAISS
from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from milvus_lite.index import load_index_from_spec

# 1. 配置全局嵌入模型
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# 2. 创建示例文档
texts = [
    "张三是法外狂徒",
    "LlamaIndex是一个用于构建和查询私有或领域特定数据的框架。",
    "它提供了数据连接、索引和查询接口等工具。"
]
docs = [Document(text=t) for t in texts]

# 3. 创建索引并持久化到本地
index = VectorStoreIndex.from_documents(docs)
persist_path = "./llamaindex_index_store"
index.storage_context.persist(persist_dir=persist_path)
print(f"LlamaIndex 索引已保存至: {persist_path}")

from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. 配置 Embedding 模型
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-zh-v1.5"
)

# 2. 加载已经保存的索引
storage = StorageContext.from_defaults(
    persist_dir="./llamaindex_index_store"
)

index = load_index_from_storage(storage)

# 3. 创建检索器
retriever = index.as_retriever(similarity_top_k=2)

# 4. 查询
results = retriever.retrieve("张三是谁？")

# 5. 查看结果
for result in results:
    print(result.node.get_content())
    print(result.score)