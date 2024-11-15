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

    position: str = Field(description="Role Worked At")
    performed_for: Optional[str] = Field(
        default=None, description="Name of the Company or Personal"
    )
    roles: Optional[List[str]] = Field(
        default=None, description="Detailed Roles and Tasks Performed"
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
    completed_at: Optional[str] = Field(default=None, description="Completion Date")
    platform_name: str = Field(description="Name of Platform")

    @validator("completed_at", pre=True, always=True)
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
    past_experience: Optional[List[PastExperience]] = Field(
        description="List of Experience Including Projects"
    )
    education_info: Optional[List[Education]] = Field(description="Education Info")
    contact_info: Contact = Field(description="Contact Information")
    certifications: Optional[List[Certification]] = Field(description="Certifications")
    awards: Optional[List[Awards]] = Field(description="List of Awards")


class State(MessagesState):
    resume: Resume


llm_with_json = llm.with_structured_output(Resume)

builder = StateGraph(State)


def experience_genrater_node(state: MessagesState):
    system_messages = """
        You are the AI Agent that Generate Experience According to Job Title and Job Description"
        'Use valid company names. If a specific company name cannot be determined, use "Personal Projects" or a similar valid designation. Also Analyze according to resume Target Date and Add it',
        "Genrate 4 to 5 Experience According to Job Title
        """

    system_messages = ("system", system_messages)
    messages = [system_messages] + state["messages"]
    response = llm.invoke(messages)
    return {"experience": response.content}


def llm_node(state: MessagesState):
    system_message = (
        "system",
        """ 
You are an AI assistant specialized in creating and updating professional resumes. Your tasks are as follows:

1. **Analyze the Existing Resume**: Carefully review the user's current resume to understand their qualifications, experience, and skills.

2. **Understand Job Requirements**: Consider the provided Job Title and Job Description to tailor the resume accordingly.

3. **Update the "About" Section**: Modify the "About" section to align with the Job Title and Job Description, highlighting relevant strengths and experiences.

4. **Add Relevant Experiences**:
   - Add new experiences that are pertinent to the Job Title and Job Description.
   - Use valid company names. If a specific company name cannot be determined, use "Personal Projects" or a similar valid designation.
   - Ensure that existing experiences are not omitted or hidden.

5. **Enhance Skills Section**:
   - Add new skills that are relevant to the Job Title and Job Description.
   - Retain the user's existing skills without omission.

6. **Optimize for Applicant Tracking Systems (ATS)**:
   - Integrate keywords from the Job Title and Job Description throughout the resume to improve ATS compatibility.
   - Ensure keywords are used naturally and contextually within the resume content.

7. **Maintain Clarity and Professionalism**:
   - Ensure the resume remains clear, concise, and professional.
   - Avoid adding any commentary or text outside of the resume content.

**Output**:
Generate an updated resume that incorporates all the above requirements. The resume should be well-structured, highlighting the user's qualifications, updated "About" section, added experiences and skills relevant to the Job Title and Job Description, and optimized with appropriate keywords for ATS tracking.

""",
    )

    llm_with_json = llm.with_structured_output(Resume)
    experience_message = ("user", f"Also Add This Experience {state["experience"]}")
    messages = [system_message] + state["messages"] + [experience_message]

    return {"messages": messages, "resume": llm_with_json.invoke(messages)}


builder.add_node("llm_node", llm_node)

builder.add_edge(START, "llm_node")
builder.add_edge("llm_node", END)

graph = builder.compile()
