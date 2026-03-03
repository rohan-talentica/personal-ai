"""
Basic conversational chain.

Wraps a ChatPromptTemplate + LLM into a reusable LCEL chain.
The chain is stateless; callers are responsible for passing message history
when they want multi-turn behaviour.
"""
from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable

from src.utils.llm import get_llm

_SYSTEM_PROMPT = (
    "You are a helpful, concise AI research assistant. "
    "Answer questions accurately and, where relevant, suggest follow-up topics."
)


def build_chat_chain(
    model: str = "openai/gpt-3.5-turbo",
    temperature: float = 0.7,
    system_prompt: str = _SYSTEM_PROMPT,
) -> Runnable:
    """Build and return a stateless conversational chain.

    The returned chain expects a dict with:
        - ``question`` (str): the user's current message
        - ``history``  (list[BaseMessage], optional): prior turn messages

    Returns a plain ``str`` answer.

    Args:
        model: OpenRouter model identifier.
        temperature: Sampling temperature.
        system_prompt: System message prepended to every conversation.
    """
    llm = get_llm(model=model, temperature=temperature)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history", optional=True),
            ("human", "{question}"),
        ]
    )

    chain: Runnable = prompt | llm | StrOutputParser()
    return chain
