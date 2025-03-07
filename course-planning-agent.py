
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain import hub
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor

from dotenv import load_dotenv
load_dotenv()

def create_retriever(urls: list[str]):
    """
    NOTE: Currently documents are just scraped webpages, would be replaced with clean tabular data from an api, or vector representations of courses and instructors can be generated from other models
    """
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
    response = agent_executor.invoke({"input": user_input}, config={"configurable": {"session_id": "123"}})
    return response["output"] + "\n"
    
def main():
    course_info_retriever = create_retriever(["https://classes.berkeley.edu/search/class?f[]=term:8558&f[]=subject_area:5417", "https://classes.berkeley.edu/search/class?f[]=term:8558&f[]=subject_area:5439"])
    course_info_retriever_tool = create_retriever_tool(course_info_retriever, "Course_information", "Search for information about courses")
    instructor_retriever = create_retriever(["https://www.ratemyprofessors.com/professor/706268", "https://www.ratemyprofessors.com/professor/16984"])
    instructor_info_retriever_tool = create_retriever_tool(instructor_retriever, "Instructor_lookup", "Use this tool to get information about instructors or professors")
    search = TavilySearchResults(max_results=2)
    
    # Set tools here - retreivers for course info and instructor info and external search  
    tools = [course_info_retriever_tool, instructor_info_retriever_tool, search]
    
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    prompt = hub.pull("hwchase17/openai-functions-agent")
    memory = ChatMessageHistory(session_id="123")
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools,verbose=False)
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: memory,
        input_messages_key="input",
        history_messages_key="chat_history"
    )

    print("Hello, feel free to ask me questions about your course planning!")
    while True:
        user_input = input()
        if user_input.lower() in ["exit", "quit"]:
            print("Have a nice day!")
            break
        try:
            agent_response = query_agent(user_input, agent_with_chat_history)
            print(agent_response)
        except Exception as err:
            # log exception
            print("Sorry we're currently unavailable at the moment!  Please try again later")

if __name__== "__main__":
    main()