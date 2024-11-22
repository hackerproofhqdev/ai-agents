from langgraph.graph import MessagesState , START , END , StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel , Field
from langchain_core.messages import HumanMessage , SystemMessage

class QuestionModel(BaseModel):
    """Question Model"""
    question : str = Field(description="Quiz Question")
    answer : str = Field(description="Correct answer of the question")
    detail :  str = Field(description="Detail Description of why this answer is correct")
    options : list[str] = Field(description="Options to choose from")

class QuizModel(BaseModel):
    """Quiz Model"""
    questions : list[QuestionModel] = Field(description="Quiz Questions")



class State(MessagesState):
  quiz : list[QuestionModel]
  past_question_summary:str



llm = ChatOpenAI(model_name="gpt-4o-mini" , temperature=0.5) # type: ignore

structured_llm = llm.with_structured_output(QuizModel)

builder = StateGraph(State)



def generate_summary_of_quiz(state: State):
    summary = state.get("past_question_summary", "")
    quiz = state.get("quiz", [])
    if not summary:
        human_message = HumanMessage(
            content=f"Generate a summary for each of the following questions in the format:\n\n"
                    f"**Question [number]**:\n\n"
                    f"Question: [question]\n"
                    f"Answer: [correct answer]\n"
                    f"Options:\n"
                    f"1 - Option 1\n"
                    f"2 - Option 2\n"
                    f"3 - Option 3\n"
                    f"4 - Option 4\n\n"
                    f"Here are the questions:\n{quiz}"
            )
        response = llm.invoke([human_message])
        summary = response.content # type: ignore
    else:
        human_message = HumanMessage(
            content=f"Here is the existing summary:\n\n{summary}\n\n"
                    f"Extend this summary by adding the following questions in the same format. "
                    f"Ensure that the final summary includes both the previous and new questions:\n\n{quiz}"
            )
        response = llm.invoke([human_message])
        summary = response.content # type: ignore
    return {"past_question_summary": summary}





def generate_quiz(state:State):
  summary = state.get("past_question_summary" , "")
  system_prompt = f"Generate Unique and Techinal and Logical Quiz According to the job title and job description . Make Sure Donot repeate the Question"
  system_message = SystemMessage(content=system_prompt)
  if not summary: 
    response  = structured_llm.invoke([system_message] + state.get("messages")) # type: ignore
  
  else:
    human_message = HumanMessage(content=f"Here is the summary of perivous questions : {summary}")
    response  = structured_llm.invoke([system_message , human_message] + state.get("messages")) # type: ignore
  return {"quiz":response.questions} # type: ignore



builder.add_node("summary_agent" , generate_summary_of_quiz)
builder.add_node("quiz_agent" , generate_quiz)

builder.add_edge(START , "quiz_agent")
builder.add_edge("quiz_agent","summary_agent")
builder.add_edge("summary_agent" , END)


graph = builder.compile()