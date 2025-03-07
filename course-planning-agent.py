
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain import hub
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor

from dotenv import load_dotenv
load_dotenv()

def create_retriever(urls: list[str]):
    docs = []
    for url in urls:
        loader = WebBaseLoader(url)
        docs.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vector = FAISS.from_documents(documents, embeddings)
    retriever = vector.as_retriever()
    return retriever

def query_agent(user_input: str, agent_executor):
    response = agent_executor.invoke({"input": user_input})
    return response["output"] + "\n"
    
def main():
    course_info_retriever = create_retriever(["https://classes.berkeley.edu/search/class?f[]=term:8558&f[]=subject_area:5417", "https://classes.berkeley.edu/search/class?f[]=term:8558&f[]=subject_area:5439"])
    course_info_retriever_tool = create_retriever_tool(course_info_retriever, "Course_information", "Search for information about courses")
    instructor_retriever = create_retriever(["https://www.ratemyprofessors.com/professor/706268", "https://www.ratemyprofessors.com/professor/16984"])
    instructor_info_retriever_tool = create_retriever_tool(instructor_retriever, "Instructor_lookup", "Use this tool to get information about instructors or professors")
    search = TavilySearchResults(max_results=2)
    tools = [course_info_retriever_tool, instructor_info_retriever_tool, search]
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    prompt = hub.pull("hwchase17/openai-functions-agent")
    memory = ConversationBufferMemory(memory_key="chat_history")
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, checkpointer=memory, verbose=True)

    print("Hello, feel free to ask me questions about your course planning!")
    while True:
        user_input = input()
        if user_input.lower() in ["exit", "quit"]:
            print("Have a nice day!")
            break
        print(query_agent(user_input, agent_executor))

if __name__== "__main__":
    main()