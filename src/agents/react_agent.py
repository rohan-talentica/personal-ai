"""
ReAct / Tool-Calling agent builder.

Uses ``langchain_classic`` (which ships ``create_tool_calling_agent`` +
``AgentExecutor``) to build an agent that can invoke the custom tools
from ``src.tools``.

Pattern from Day 5 notebook (day5_react_agents_tools.ipynb).
"""
from __future__ import annotations

from typing import Any

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool

from src.tools.custom_tools import ALL_TOOLS
from src.utils.llm import get_llm

_SYSTEM_PROMPT = """\
You are a helpful AI research assistant with access to several tools.

Think step-by-step:
1. Decide whether you need a tool or can answer directly.
2. If you need a tool, call it with the correct input.
3. Observe the result and incorporate it into your final answer.
4. Provide a concise, accurate final answer.

Always explain your reasoning briefly before using a tool.
"""


def build_react_agent(
    tools: list[BaseTool] | None = None,
    model: str = "openai/gpt-3.5-turbo",
    temperature: float = 0.0,
    max_iterations: int = 6,
    verbose: bool = False,
) -> AgentExecutor:
    """Build and return a tool-calling AgentExecutor.

    The agent has access to ``calculator``, ``word_counter``,
    ``get_weather``, and ``text_summarizer`` by default.  Pass a
    custom ``tools`` list to override.

    Args:
        tools: Override the default tool list.
        model: OpenRouter model identifier.
        temperature: Sampling temperature (0 recommended for agents).
        max_iterations: Maximum tool-call loop iterations before stopping.
        verbose: Whether to print the agent's thought process to stdout.

    Returns:
        A configured ``AgentExecutor`` instance.
    """
    if tools is None:
        tools = ALL_TOOLS

    llm = get_llm(model=model, temperature=temperature)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=max_iterations,
        verbose=verbose,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )
    return executor


def run_agent(query: str, agent: AgentExecutor | None = None, **kwargs: Any) -> dict[str, Any]:
    """Convenience wrapper — build (or reuse) an agent and run it.

    Args:
        query: The user's question or instruction.
        agent: Pre-built ``AgentExecutor``. Built fresh each call if None.
        **kwargs: Extra keyword arguments forwarded to ``AgentExecutor.invoke``.

    Returns:
        A dict with ``output`` (str) and ``steps`` (list of intermediate steps).
    """
    if agent is None:
        agent = build_react_agent()

    result = agent.invoke({"input": query}, **kwargs)
    return {
        "output": result.get("output", ""),
        "steps": [
            {
                "tool": step[0].tool,
                "tool_input": step[0].tool_input,
                "observation": step[1],
            }
            for step in result.get("intermediate_steps", [])
        ],
    }
