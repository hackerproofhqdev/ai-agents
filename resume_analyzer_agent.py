from typing import TypedDict
from langgraph.graph import START, END, StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini")


class State(TypedDict):
    resume: str
    job_title: str
    job_desc: str
    suggestions: str


builder = StateGraph(State)


def resume_analyzer(state: State):
    messages = [
        (
            "system",
            "You are an AI agent specialized in analyzing user resumes. Your task is to match the user's resume with a given job title and job description. Highlight any missing points, suggest additions, and explain them thoroughly to help the user improve their resume.",
        ),
        (
            "human",
            """Please analyze my resume according to the following job title and job description. Highlight any missing points, suggest additions, and explain them thoroughly.

**My Resume**:
{resume}

**Job Title**:
{job_title}


**Job Description**:
{job_desc}

**Instructions**:
- Focus on key skills, experiences, and qualifications required for the job.
- Provide feedback in bullet points for clarity.
- Suggest specific changes or additions I can make to my resume.
""",
        ),
    ]
    prompt_template = ChatPromptTemplate.from_messages(messages)
    chain = prompt_template | llm
    response = chain.invoke(
        {
            "resume": state["resume"],
            "job_title": state["job_title"],
            "job_desc": state["job_desc"],
        }
    )
    return {"suggestions": response.content}


builder.add_node("assistant", resume_analyzer)


builder.add_edge(START, "assistant")
builder.add_edge("assistant", END)

graph = builder.compile()
