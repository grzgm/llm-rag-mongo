""" Custom prompt template
"""
from langchain_core.prompts import ChatPromptTemplate

TEMPLATE = """Answer the question based only on the following context:
{context}

Question: {question}
"""

PROMPT = ChatPromptTemplate.from_template(TEMPLATE)
