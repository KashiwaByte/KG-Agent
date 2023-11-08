# 工具集
# Tool webloader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders import TextLoader
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.tools import BaseTool
import gradio as gr


    
llm = OpenAI(temperature=0)
loader = WebBaseLoader("https://beta.ruff.rs/docs/faq/")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

 
embeddings = OpenAIEmbeddings()
docs = loader.load()
ruff_texts = text_splitter.split_documents(docs)
ruff_db = Chroma.from_documents(ruff_texts, embeddings, collection_name="ruff")
ruff = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=ruff_db.as_retriever())



# Tool search
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
search = SerpAPIWrapper(serpapi_api_key="put your key here")
search_tool = Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    )





llm = OpenAI(temperature=0)


tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    ),
    Tool(
        name = "Ruff QA System",
        func=ruff.run,
        description="useful for when you need to answer questions about ruff (a python linter). Input should be a fully formed question."
    ),
]

# Construct the agent. We will use the default agent type here.
# See documentation for a full list of options.
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

#询问指定文件内容
agent.run("介绍一下Ruff?")

def answer(ask):
    ans=agent.run(ask)
    return ans



ui = gr.Interface(fn=answer, inputs=gr.Textbox(label="请输入你想查询的信息："), outputs="text",title = "<h1 style='font-size: 40px;'><center>知识图谱问答</center></h1>")
ui.launch()
