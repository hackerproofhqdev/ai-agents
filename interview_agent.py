from langgraph.graph import StateGraph , MessagesState , START , END
from langchain_openai import ChatOpenAI


llm = ChatOpenAI(model="gpt-4o-mini")



builder = StateGraph(MessagesState)



def interview_agent(state:MessagesState):
  system_prompt = "You are an interviewer. Analyze the provided job topic and conduct an interview with the user based on it. Ask one question at a time, as professional interviewers do. You may conclude the interview when you feel it is sufficient . Also Provide How Much Marks You Will give to User"
  system_message = ("system"  , system_prompt)
  messages = [system_message] + state["messages"]
  return {"messages": [llm.invoke(messages)]}



builder.add_node("interview_agent" , interview_agent)
builder.add_edge(START , "interview_agent")
builder.add_edge("interview_agent" , END)


graph = builder.compile()