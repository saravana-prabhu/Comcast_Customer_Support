from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st
import pinecone
from langchain.vectorstores import Pinecone

# session_state key for storing messages
msgs = StreamlitChatMessageHistory(key="customer_messages")

#initializing keys
openai_key = st.secrets["openai_key"]
pinecone_key = st.secrets["pinecone_key"]


pinecone.init(api_key= pinecone_key, environment="gcp-starter")
embed = OpenAIEmbeddings(openai_api_key= openai_key)
db = Pinecone.from_existing_index("comcast", embed)
llm = ChatOpenAI(openai_api_key= openai_key)
retriever = db.as_retriever()

class ComcastAgent:
    
    def __init__(self,verbose: bool=True) -> None:
        tool = create_retriever_tool(
            retriever,
            "search_comcast_customer_queries",
            "Searches and returns documents regarding customer queries.",
        )
        tools = [tool]

        # This is needed for both the memory and the prompt
        memory_key = "history"
        memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm,chat_memory=msgs, max_token_limit=10000)
        system_message = SystemMessage(
            content=(
                """
"Saravana Customer Support" role at Comcast Business: chat-based help, maintain satisfaction, avoid irritation, professional, user-friendly, don't disclose AI identity, comprehensive Comcast knowledge crucial. Context: Comcast Business (Comcast subsidiary) provides internet, phone, cable TV solutions to businesses. You: experienced agent, deep company understanding, professional, approachable, clean, well-groomed. Traits: patient, empathetic, knowledgeable, excellent communicator, problem-solver, friendly, adaptable, tech-savvy, organized, positive. Goal: End interactions positively, address/resolve issues promptly, leave customers valued, satisfied, meet/exceed satisfaction metrics. Responses: courteous, informative, empathetic, reflect Comcast's commitment to excellence. If unsure, say "don't know", guide to resources. Use tools for info lookup, respond in user's language."""
            )
            

        )
        prompt = OpenAIFunctionsAgent.create_prompt(
            system_message=system_message,
            extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)],
        )
        agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True,
            return_intermediate_steps=True,
        )
        
    def get_answer(self, query: str) -> str:
        result = self.agent_executor({"input": query})
        return result["output"]
    
