import os
from pathlib import Path

# 模型已经下载到本地，运行时不再访问 Hugging Face/Xet。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

project_root = Path(__file__).resolve().parents[2]
markdown_path = project_root / "data/C1/markdown/easy-rl-chapter1.md"
model_path = Path.home() / "models/bge-small-zh-v1.5"

if not model_path.is_dir():
    raise FileNotFoundError(f"本地嵌入模型不存在：{model_path}")

# 加载本地markdown文件
loader = UnstructuredMarkdownLoader(markdown_path)
docs = loader.load()

# 文本分块
text_splitter = RecursiveCharacterTextSplitter()
chunks = text_splitter.split_documents(docs)
print("分片总数：",len(chunks))

# 中文嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name=str(model_path),
    model_kwargs={'device': 'cpu', 'local_files_only': True},
    encode_kwargs={'normalize_embeddings': True}
)
  
# 构建向量存储
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(chunks)

# 提示词模板
prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”

上下文:
{context}

问题: {question}

回答:"""
                                          )

llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=4096,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 用户查询
question = "文中举了哪些例子？"

# 在向量存储中查询相关文档
retrieved_docs = vectorstore.similarity_search(question, k=3)
for i,doc in enumerate(retrieved_docs):
    print(f"Chunk {i + 1}，长度：{len(doc.page_content)}")
    print(doc.page_content)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

answer = llm.invoke(prompt.format(question=question, context=docs_content))
print(answer.content)
