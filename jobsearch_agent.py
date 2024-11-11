from typing import List, Literal, Optional ,Any, Union
from langgraph.graph import START , END  , StateGraph , MessagesState
from langgraph.prebuilt import ToolNode 
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AnyMessage ,SystemMessage 
from langchain_openai import ChatOpenAI
from  pydantic import BaseModel , Field


class Job(BaseModel):
    """Job Model"""
    company_name: str = Field(description="Company Name")
    role: str = Field(description="Role They Are looking For")
    working_exp: str = Field(description="Working Experience they Required")
    type: Literal["onsite", "remote", "hybrid"] = Field(description="Job Type")
    desc: str = Field(description="Some More Description")
    url_to_apply : str = Field(description="Url to Apply")
    # Additional fields
    location: Optional[str] = Field(default=None, description="Location of the Job")
    salary_range: Optional[str] = Field(default=None, description="Salary Range Offered")
    application_deadline: Optional[str] = Field(default=None, description="Deadline for Application")
    skills_required: List[str] = Field(default_factory=list, description="List of Skills Required")
    benefits: Optional[List[str]] = Field(default=None, description="Job Benefits")
    contact_email: Optional[str] = Field(default=None, description="Contact Email for Applications")
    employment_type: Optional[Literal["full-time", "part-time", "contract", "temporary"]] = Field(
        default=None, description="Type of Employment"
    )

class Jobs(BaseModel):
    "Jobs Model"
    jobs : list[Job]
    
class State(MessagesState):
    jobs : list[Job]

tavily_client = TavilySearchResults(include_answer=True)


tools = [tavily_client]

def query_llm(state:State):
    llm = ChatOpenAI(model="gpt-4o-mini")
    system_message = SystemMessage(content="You are an AI agent that will search every query on the internet and generate the response.")
    llm_with_tools = llm.bind_tools(tools)
    messages = [system_message] + state["messages"]
    return {"messages" : [llm_with_tools.invoke(messages)]}


def format_llm(state:State):
   llm = ChatOpenAI(model="gpt-4o-mini")
   llm_with_structued_output = llm.with_structured_output(Jobs)
   return {"jobs" : llm_with_structued_output.invoke(state["messages"][-1].content)}




builder = StateGraph(State)


tool_node = ToolNode(tools=tools)

def tool_condition(
    state: Union[list[AnyMessage], dict[str, Any], BaseModel],
    messages_key: str = "messages",
) -> Literal["tools", "format_llm"]:
    if isinstance(state, list):
        ai_message = state[-1]
    elif isinstance(state, dict) and (messages := state.get(messages_key, [])):
        ai_message = messages[-1]
    elif messages := getattr(state, messages_key, []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "format_llm"


builder.add_node("query_llm" , query_llm)
builder.add_node("format_llm" , format_llm)
builder.add_node("tools" , tool_node)

builder.add_edge(START , "query_llm")
builder.add_conditional_edges(
    "query_llm",
    tool_condition
)
builder.add_edge("tools" , "query_llm")
builder.add_edge("query_llm" , 'format_llm')
builder.add_edge('format_llm' , END)

graph = builder.compile()