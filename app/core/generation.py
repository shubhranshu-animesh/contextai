from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.config import settings

class GenerationEngine:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0.3, # Slightly higher temp for better general chat
            groq_api_key=settings.GROQ_API_KEY, 
            model_name="llama-3.1-8b-instant" 
        )

        # PROMPT 1: Strict RAG Mode
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are CONTEXTai, an elite intelligence assistant. 
            Your core directive is to answer the user's question based ONLY on the provided context.
            
            Strict Operational Rules:
            1. If the answer is not contained within the context, you must output: "I cannot answer this based on the provided documents."
            2. Do not use outside knowledge or pre-trained memory.
            3. Be precise, professional, and concise.
            
            Previous Conversation History:
            {chat_history}
            
            Context provided from documents:
            {context}"""),
            ("human", "{question}")
        ])

        # PROMPT 2: General Chat Mode
        self.general_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are CONTEXTai, a helpful and intelligent AI assistant. 
            You are currently in General Chat Mode because the user has not uploaded a document yet.
            Answer their questions thoughtfully and naturally using your general knowledge.
            
            Previous Conversation History:
            {chat_history}"""),
            ("human", "{question}")
        ])

    def stream_answer(self, query: str, context: str, history: list, use_rag: bool = True):
        """Streams the AI response, dynamically choosing the prompt based on use_rag flag."""
        
        formatted_history = ""
        for msg in history:
            formatted_history += f"{msg.role.capitalize()}: {msg.content}\n"
            
        if not formatted_history:
            formatted_history = "No previous conversation."

        # Dynamically select the chain and inputs
        if use_rag:
            chain = self.rag_prompt | self.llm | StrOutputParser()
            inputs = {"context": context, "question": query, "chat_history": formatted_history}
        else:
            chain = self.general_prompt | self.llm | StrOutputParser()
            inputs = {"question": query, "chat_history": formatted_history}

        for chunk in chain.stream(inputs):
            if hasattr(chunk, 'content'):
                yield chunk.content
            else:
                yield str(chunk)