from langgraph.graph import StateGraph , MessagesState , START , END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


llm = ChatOpenAI(model="gpt-4o-mini")

class StateMessages(MessagesState):
    job_title:str
    job_desc:str

builder = StateGraph(StateMessages)

def llm_node(state: StateMessages):
    messages = [
        (
            "system",
            (
                "You are an AI assistant specialized in analyzing job postings. "
                "Your task is to provide a comprehensive analysis of a given job description. "
                "The analysis should include the following sections:\n"
                "1. **Key Responsibilities**\n"
                "2. **Required Skills**\n"
                "3. **Educational Qualifications**\n"
                "4. **Potential Career Paths**\n\n"
                "Please ensure the response is well-structured, concise, and formatted in markdown."
            ),
        ),
        (
            "human",
            (
                "### Job Details\n"
                "**Job Title:** {title}\n"
                "**Job Description:**\n{desc}\n\n"
                "### Analysis\n"
                "Provide a detailed analysis based on the above job details."
            ),
        ),
    ]
    prompt_template = ChatPromptTemplate.from_messages(messages)
    chain = prompt_template | llm
    return {
        "messages": [
            chain.invoke(
                {"title": state.get("job_title"), "desc": state.get("job_desc")}
            )
        ]
    }


builder.add_node("llm_node" , llm_node)

builder.add_edge(START , "llm_node")
builder.add_edge("llm_node" , END)

graph = builder.compile()