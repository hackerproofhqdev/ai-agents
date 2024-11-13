from typing import Optional
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


llm = ChatOpenAI(model="gpt-4o-mini")


class Skills(BaseModel):
    """Skills Model"""

    skill_name: str = Field(description="Skill Name")
    description: str = Field(description="Skill Description")


class PastExperience(BaseModel):
    """PastExperience Model"""

    position: str = Field(description="Role Worked At")
    performed_for: Optional[str] = Field(
        default=None, description="Name of the Company or Personal"
    )
    roles: Optional[list[str]] = Field(
        default=None, description="Detailed roles and various tasks performed"
    )
    start_date: str = Field(description="Start Date")
    end_date: str = Field(description="End Date")


class Contact(BaseModel):
    """Contact Info Model"""

    phone_number: str = Field(description="Phone Number of Resumer")
    email: str = Field(description="Email of Resumer")


class Education(BaseModel):
    """Education Model"""

    degree: str = Field(description="Degree Details")
    completed_at: str = Field(description="Completion Date")
    platform_name: str = Field(description="Name of Platform")


class Certification(BaseModel):
    """Certification Model"""

    name: str = Field(description="Name of the Certification")
    issued_date: Optional[str] = Field(description="Date of Issue")


class Award(BaseModel):
    """Award Model"""

    title: str = Field(description="Award Title")


class Resume(BaseModel):
    """Resume Model"""

    name: str = Field(description="Name of the Resumer")
    role: str = Field(description="Role of the Resumer")
    about: str = Field(description="About the Person")
    skills: list[Skills] = Field(description="List of Skills")
    past_experience: Optional[list[PastExperience]] = Field(
        default=None, description="List of Experiences Including Projects"
    )
    education_info: Optional[list[Education]] = Field(default=None, description="Education Information")
    contact_info: Contact = Field(description="Contact Information")
    certifications: Optional[list[Certification]] = Field(default=None, description="Certifications")
    awards: Optional[list[Award]] = Field(default=None, description="List of Awards")


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
