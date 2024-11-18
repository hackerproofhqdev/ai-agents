from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from pydantic import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode

# from langchain_core.messages import AnyMessage
# from typing import Any , Union , Literal

search_tool = TavilySearchResults(max_results=10)


class CoverLetterModel(BaseModel):
    """Cover Letter Model"""

    introduction: str = Field(description="Intrduction Of Resume")
    experiences: str = Field(description="Experience Section")
    why_chosse_me: str = Field(description="Why Chosse Me Section")
    qualifications: str = Field(description="Qualifications Section")
    alignment_with_role: str = Field(description="Alignment With Role Section")
    call_to_action: str = Field(description="Call To Action Section")
    closing: str = Field(description="Closing Section")


llm = ChatOpenAI(model="gpt-4o-mini", verbose=True)


class State(MessagesState):
    cover_letter_text: str
    cover_letter: CoverLetterModel
    job_analysis: str


builder = StateGraph(State)


tools = [search_tool]


def keyword_analyzer_agent(state: State):
    system_message = "You are the AI Agent responsible for Identifies key responsibilities, required skills, and qualifications from the job description."
    messages = [("system", system_message)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": response}


def reasearch_agent(state: State):
    llm_with_tools = llm.bind_tools(tools)
    system_message = "You are AI Agent responsible for Extracts details about the company’s values, culture, and mission."
    messages = [("system", system_message)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": response}


def analyzer_agent(state: State):
    system_message = "You are AI Agent responsible for merging all analysis of job done by other agents."
    messages = [("system", system_message)] + state["messages"]
    response = llm.invoke(messages)
    return {"job_analysis": response.content}


def content_drafting_agent(state: State):
    system_prompr = "You are the AI Agent Responsible For Creating the cover letter according to the job analysis (done by analysis agent) and resume."
    messages = [("system", system_prompr), ("ai", state["job_analysis"])]
    response = llm.invoke(messages)
    return {"messages": response}


def editing_agent(state: State):
    system_prompt = """
  You are a professional editor specializing in job application materials. Review the provided cover letter draft and make improvements focusing on:

Clarity and Conciseness: Ensure all points are clearly expressed without unnecessary words.
Tone and Voice: Maintain a professional, confident, and sincere tone appropriate for the industry and company culture.
Logical Flow: Improve transitions between ideas and ensure the letter has a coherent structure.
Impactful Language: Use strong, action-oriented language that highlights the candidate's value.
  """
    messages = [("system", system_prompt)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": response}


def proofreading_agent(state: State):
    sytem_message = """
  You are an experienced proofreader with a meticulous eye for detail. Carefully examine the cover letter for:

Grammar and Punctuation: Correct any grammatical errors and ensure proper punctuation.
Spelling: Fix any spelling mistakes.
Formatting: Ensure consistent font usage, alignment, and overall presentation.
Adherence to Standards: Confirm the letter meets professional standards for cover letters.
  """
    messages = [("system", sytem_message)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": response}


def formating_agent(state: State):
    system_message = """
  Convert the cover letter into the format .
  The `CoverLetterModel` is a structured representation of a professional cover letter, designed to standardize its content and format. It consists of the following sections:

1. **Introduction**: A brief opening statement that introduces the candidate and sets the tone for the cover letter.
2. **Experiences**: Highlights the candidate's relevant work experiences and achievements that align with the role.
3. **Why Choose Me**: A persuasive section explaining why the candidate is the best fit for the position.
4. **Qualifications**: Details the candidate’s educational background, certifications, and other qualifications.
5. **Alignment with Role**: Demonstrates how the candidate's skills and experiences align with the requirements of the role.
6. **Call to Action**: Encourages the hiring manager to take the next step, such as arranging an interview.
7. **Closing**: A professional closing statement to end the cover letter on a strong note.

Each section is represented as a string field with a clear description, ensuring that all aspects of a well-crafted cover letter are systematically addressed.
  """
    messages = [("system", system_message)] + state["messages"]
    response = llm.invoke(messages)
    return {"cover_letter_text": response.content}


def cover_letter_agent(state: State):
    structured_llm = llm.with_structured_output(CoverLetterModel)
    system_message = "Convert This Follwing Text into structred cover letter"
    messages = [("system", system_message), ("ai", state["cover_letter_text"])]
    response = structured_llm.invoke(messages)
    return {"cover_letter": response}


# def tool_condition(
#     state: Union[list[AnyMessage], dict[str, Any], BaseModel],
#     messages_key: str = "messages",
# ) -> Literal["tools", "analyzer_agent"]:
#     if isinstance(state, list):
#         ai_message = state[-1]
#     elif isinstance(state, dict) and (messages := state.get(messages_key, [])):
#         ai_message = messages[-1]
#     elif messages := getattr(state, messages_key, []):
#         ai_message = messages[-1]
#     else:
#         raise ValueError(f"No messages found in input state to tool_edge: {state}")
#     if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
#         return "tools"
#     return "analyzer_agent"

tool_node = ToolNode(tools=tools)

builder.add_node("keyword_analyzer_agent", keyword_analyzer_agent)
builder.add_node("reasearch_agent", reasearch_agent)
builder.add_node("analyzer_agent", analyzer_agent)
builder.add_node("tools", tool_node)
builder.add_node("content_drafting_agent", content_drafting_agent)
builder.add_node("editing_agent", editing_agent)
builder.add_node("proofreading_agent", proofreading_agent)
builder.add_node("formating_agent", formating_agent)
builder.add_node("cover_letter_agent", cover_letter_agent)


builder.add_edge(START, "keyword_analyzer_agent")
builder.add_edge("keyword_analyzer_agent", "reasearch_agent")
builder.add_edge("reasearch_agent", "tools")
builder.add_edge("tools", "analyzer_agent")
# builder.add_edge("reasearch_agent" , "analyzer_agent")
builder.add_edge("analyzer_agent", "content_drafting_agent")
builder.add_edge("content_drafting_agent", "editing_agent")
builder.add_edge("editing_agent", "proofreading_agent")
builder.add_edge("proofreading_agent", "formating_agent")
builder.add_edge("formating_agent", "cover_letter_agent")
builder.add_edge("cover_letter_agent", END)

graph = builder.compile()
