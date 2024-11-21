from typing import List, Optional
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, validator

llm = ChatOpenAI(model="gpt-4o-mini")


class Skills(BaseModel):
    """Skills Model"""

    skill_name: str = Field(description="Skill Name")
    description: str = Field(description="Skill Description")


class PastExperience(BaseModel):
    """PastExperience Model"""

    position: str = Field(description="Role Work At")
    performed_for: Optional[str] = Field(
        default=None, description="Name Of the Company or Personal"
    )
    roles: Optional[List[str]] = Field(
        default=None, description="Detail Roles and everything performed"
    )
    start_date: Optional[str] = Field(default=None, description="Date Of Started")
    end_date: Optional[str] = Field(default=None, description="Date of End")


class Contact(BaseModel):
    """Contact Info Model"""

    phone_number: str = Field(description="Phone Number of Resumer")
    email: str = Field(description="Email of Resumer")


class Education(BaseModel):
    """Education Model"""

    degree: str = Field(description="Degree Details")
    completed_at: Optional[str] = Field(default=None, description="Completion Date")
    platform_name: str = Field(description="Name of Platform")

    @validator("completed_at", pre=True, always=True)
    def set_completed_at(cls, v):
        return v or "Date Not Provided"


class Certification(BaseModel):
    """Certification Model"""

    name: str = Field(description="Name of the Certification")
    issued_at: Optional[str] = Field(default=None, description="Date of Issue")


class Projects(BaseModel):
    project_name: str = Field(description="Project Name")
    description: str = Field(
        description="Explanation of How he Performed His Role What Work He have Done"
    )


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
    projects: list[Projects] = Field(description="Projet List")
    awards: Optional[list[Awards]] = Field(description="List Of Awards")


class ListExperience(BaseModel):
    experiences: list[PastExperience]


class ListProjects(BaseModel):
    projects: list[Projects]


class State(MessagesState):
    resume: Resume
    about:str
    experience: List[PastExperience]
    projects: list[Projects]
    experience: list[PastExperience]


llm_with_json = llm.with_structured_output(Resume)

builder = StateGraph(State)


def experience_genrater_node(state: State):
    system_messages = """
    You are an AI Agent that generates experience according to Job Title and Job Description.
    Use valid company names. Avoid using "Personal Projects" or any other similar placeholders. Always prefer to generate a valid company name wherever applicable.
    Also, analyze the resume's target date and add experiences based on it.
    Also Define that how user will peformed the Role in which Projects He work All of that
    Generate 10 experiences based on the Job Title, with valid company names included in each experience.
"""

    system_messages = ("system", system_messages)
    messages = [system_messages] + state["messages"]
    structured_llm = llm.with_structured_output(ListExperience)
    response = structured_llm.invoke(messages)
    return {"experience": response.experiences}


def project_agent(state: State):
    system_message = """
ou are an AI agent specializing in generating past work experiences tailored to a specific job title. Your task is to generate relevant and realistic projects associated with the given job title.

Each project should have a valid and professional name that aligns with industry standards (avoid placeholders like 'XYZ' or 'ABC').
Provide a detailed description of each project, focusing on:
The user’s role and responsibilities within the project.
Technologies, tools, or methodologies used.
The impact or outcome of the project where applicable.
Ensure the projects are highly relevant to the job title and highlight the user's expertise effectively.
"""
    system_messages = ("system", system_message)
    messages = [system_messages] + state["messages"]
    structured_llm = llm.with_structured_output(ListProjects)
    response = structured_llm.invoke(messages)
    return {"projects": response.projects}


def profile_summary_agent(state: State):
    system_message = "You are AI Agent that generate the profile summary according to the job requriement . Build profile section according to the job title . Just Take Experience user have and add it into your"
    system_messages = ("system", system_message)
    messages = [system_messages] + state["messages"]
    response = llm.invoke(messages)
    return {"about": response.content}


def llm_node(state: State):
    system_message = (
        "system",
        """ 
You are an AI assistant specialized in creating and updating professional resumes. Your tasks are as follows:

1. **Analyze the Existing Resume**: Carefully review the user's current resume to understand their qualifications, experience, and skills.

2. **Understand Job Requirements**: Consider the provided Job Title and Job Description to tailor the resume accordingly.

3. **Update the "About" Section**: Modify the "About" section to align with the Job Title and Job Description, highlighting relevant strengths and experiences.

4. **Enhance Skills Section**:
   - Add new skills that are relevant to the Job Title and Job Description.
   - Retain the user's existing skills without omission.

5. **Update Experience Section**:
   - **Retain all existing experiences in the resume without omission or modification**.
   - **Add the new experiences provided, integrating them appropriately within the experience section**.

6. **Optimize for Applicant Tracking Systems (ATS)**:
   - Integrate keywords from the Job Title and Job Description throughout the resume to improve ATS compatibility.
   - Ensure keywords are used naturally and contextually within the resume content.

7. **Maintain Clarity and Professionalism**:
   - Ensure the resume remains clear, concise, and professional.
   - Use standard resume formatting and structure.
   - Avoid adding any commentary or text outside of the resume content.

**Output**:
Generate an updated resume that incorporates all the above requirements. The resume should be well-structured, highlighting the user's qualifications, updated "About" section, existing experiences, added new experiences, and added skills relevant to the Job Title and Job Description, optimized with appropriate keywords for ATS tracking.

""",
    )

    llm_with_json = llm.with_structured_output(Resume)
    experience_message = (
        "user",
        f"Please add the following new experiences to the resume:\n{state['experience']}",
    )
    messages = [system_message] + state["messages"] + [experience_message]
    updated_resume = llm_with_json.invoke(messages)
    return {"messages": messages, "resume": updated_resume}


builder.add_node("llm_node", llm_node)
builder.add_node("experince_agent", experience_genrater_node)
builder.add_node("project_agent", project_agent)
builder.add_node("profile_summary_agent", profile_summary_agent)
builder.add_edge(START, "experince_agent")
builder.add_edge("experince_agent", "profile_summary_agent")
builder.add_edge("profile_summary_agent", "project_agent")
builder.add_edge("project_agent", "llm_node")
builder.add_edge("llm_node", END)

graph = builder.compile()
