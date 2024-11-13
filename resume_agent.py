from typing import Optional
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


llm = ChatOpenAI(model="gpt-4o-mini")


class Skills(BaseModel):
    """Skilles Model"""

    skill_name: str = Field(description="Skill Name")
    description: str = Field(description="Skill description")


class PastExperience(BaseModel):
    """PastExperience Model"""

    postion: str = Field(description="Role Work At")
    performed_for: Optional[str] = Field(
        default=None, description="Name Of the Company or Personal"
    )
    roles: Optional[list[str]] = Field(
        default=None, description="Detail Roles and very things performed"
    )
    start_date: str = Field(description="Date Of Started")
    end_date: str = Field(description="Date of End")


class Contact(BaseModel):
    """Contact Info Model"""

    phone_number: str = Field(description="Phone Number of resumer")
    email: str = Field(description="Email of Resumer")


class Education(BaseModel):
    """Education Model"""

    degree: str = Field(description="Degress Details")
    completed_at: str = Field(description="Complete Date")
    platform_name: str = Field(description="Name Of Platform")


class Certification(BaseModel):
    """Certification Model"""

    name: str = Field(description="Name Of the Certification")
    issued_at: Optional[str] = Field(description="The Date Of Issue")


class Awards(BaseModel):
    """Award Model"""

    award: str = Field(description="Award Name")


class Resume(BaseModel):
    """Resume Model"""

    name: str = Field(description="Name of The Resumer")
    role: str = Field(description="Role of the Resumer")
    about: str = Field(description="Person About")
    skills: list[Skills] = Field(description="List of Skills")
    past_experince: Optional[list[PastExperience]] = Field(
        description="List Of Experience Including Projects"
    )
    education_info: Optional[list[Education]] = Field(description="Education Info")
    contact_info: Contact = Field(description="Contact Information")
    certifications: Optional[list[Certification]] = Field(description="Certifications")
    awards: Optional[list[Awards]] = Field(description="List Of Awards")


class State(MessagesState):
    resume: Resume


llm_with_json = llm.with_structured_output(Resume)

builder = StateGraph(State)


system_message = (
    "system",
    "You are an AI assistant specialized in creating professional resumes. Given the user's information, generate a  resume that highlights their qualifications, experience, and skills. Ensure the resume is clear and concise make sure it is also analyze By ATS Systems. Do not include any additional commentary or text outside of the resume itself. Only output the resume.",
)


def llm_node(state: State):

    messages = [system_message] + state["messages"]
    return {"messages": messages, "resume": llm_with_json.invoke(messages)}


builder.add_node("llm_node", llm_node)

builder.add_edge(START, "llm_node")
builder.add_edge("llm_node", END)

graph = builder.compile()
