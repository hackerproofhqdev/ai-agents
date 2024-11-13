from typing import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph  , START , END
from langchain_core.pydantic_v1 import BaseModel , Field
from langchain_core.prompts import ChatPromptTemplate

class Question(BaseModel):
  """Question Model"""
  question : str = Field(description="Question to be answered")
  answer : str = Field(description="Correct answer")
  options : list[str] = Field(description="Options to choose from")


class Quiz(BaseModel):
  """Quiz Model"""
  questions : list[Question] = Field(description="List of questions")


class State(TypedDict):
  """State Model"""
  quiz : Quiz
  job_description : str



def llm_node(state:State):
  llm = ChatOpenAI(model="gpt-4o-mini" , temperature=0.5)
  llm_json = llm.with_structured_output(Quiz)
  messages = [
      ("system" , "You Are the Interviwer .  You have to Take the Make  Quiz From for User according to Job Description"),
      ("human" , "Here is the Job Desc : \n {job_description}")
  ] 
  messages_template = ChatPromptTemplate.from_messages(messages)
  chain = messages_template | llm_json
  quiz = chain.invoke({"job_description":state['job_description']})
  return {"quiz":quiz}

builder = StateGraph(State)

builder.add_node("llm_node" , llm_node)
builder.add_edge(START , "llm_node")
builder.add_edge("llm_node" , END)

graph = builder.compile()