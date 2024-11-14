from typing import List, Optional
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field , validator


llm = ChatOpenAI(model="gpt-4o-mini")


class Skills(BaseModel):
    """Skills Model"""
    skill_name: str = Field(description="Skill Name")
    description: str = Field(description="Skill Description")

class PastExperience(BaseModel):
    """PastExperience Model"""
    position: str = Field(description="Role Worked At")
    performed_for: Optional[str] = Field(default=None, description="Name of the Company or Personal")
    roles: Optional[List[str]] = Field(default=None, description="Detailed Roles and Tasks Performed") 
    start_date: str = Field(description="Start Date")
    end_date: str = Field(description="End Date")

class Contact(BaseModel):
    """Contact Info Model"""
    phone_number: str = Field(description="Phone Number of Resumer")
    email: str = Field(description="Email of Resumer")

class Education(BaseModel):
    """Education Model"""
    degree: str = Field(description="Degree Details")
    completed_at: Optional[str] = Field(default=None, description="Completion Date")
    platform_name: str = Field(description="Name of Platform")

    @validator('completed_at', pre=True, always=True)
    def set_completed_at(cls, v):
        return v or "Date Not Provided"

class Certification(BaseModel):
    """Certification Model"""
    name: str = Field(description="Name of the Certification")
    issued_at: Optional[str] = Field(default=None, description="Date of Issue")

class Awards(BaseModel):
    """Award Model"""
    award: str = Field(description="Award Name")

class Resume(BaseModel):
    """Resume Model"""
    name: str = Field(description="Name of the Resumer")
    role: str = Field(description="Role of the Resumer")
    about: str = Field(description="Person About")
    skills: List[Skills] = Field(description="List of Skills")
    past_experience: Optional[List[PastExperience]] = Field(description="List of Experience Including Projects")
    education_info: Optional[List[Education]] = Field(description="Education Info")
    contact_info: Contact = Field(description="Contact Information")
    certifications: Optional[List[Certification]] = Field(description="Certifications")
    awards: Optional[List[Awards]] = Field(description="List of Awards")


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
