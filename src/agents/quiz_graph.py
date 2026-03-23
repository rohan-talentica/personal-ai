"""
LangGraph state machine for Socratic Revision quizzes.

The graph handles:
1. Generating a question based on Notion notes
2. Presenting the question to the user (interrupts execution)
3. Evaluating the user's answer
4. Recording weak areas and conditionally asking follow-ups

This runs statelessly on FastAPI but uses a checkpointer (AsyncPostgresSaver)
to restore conversation state when the user submits an answer.
"""

MAX_QUESTIONS = 3

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from src.utils.llm import get_llm

MAX_QUESTIONS = 3

class QuizState(TypedDict):
    """LangGraph state schema for Socratic revision."""
    messages: Annotated[List[AnyMessage], operator.add]
    date: str
    content: str
    questions_asked: int
    weak_areas: Annotated[List[str], operator.add]
    asked_concepts: Annotated[List[str], operator.add]  # every concept tested so far
    evaluation_feedback: str
    last_concept: str           # concept tested in the most recent evaluation
    last_confidence_score: float  # 0.0-1.0 score from the most recent evaluation
    is_completed: bool


class EvaluationOutput(BaseModel):
    """Structured LLM output for the evaluation step."""
    feedback: str = Field(description="Direct, encouraging feedback to the user on their answer.")
    is_correct: bool = Field(description="True if the user correctly grasped the core concept.")
    concept_tested: str = Field(description="The core topic or concept that was tested.")
    confidence_score: float = Field(
        description=(
            "A score from 0.0 to 1.0 reflecting how well the user demonstrated understanding. "
            "1.0 = perfect, complete answer. 0.5 = partial understanding. 0.0 = completely wrong or blank."
        )
    )


class QuestionOutput(BaseModel):
    """Structured LLM output for generating a single question."""
    question: str = Field(
        description="The exact, specific technical question to ask the user. Must ONLY be the question. Must NOT include conversational fillers, feedback, or repeat the user's previous answer."
    )


def generate_question(state: QuizState) -> Dict[str, Any]:
    """Generate a new question based on the notes. Does NOT pass message history
    to avoid the LLM feeling compelled to acknowledge prior answers."""
    llm = get_llm(use_case="quiz", temperature=0)
    question_llm = llm.with_structured_output(QuestionOutput)

    # Build an exclusion hint from concepts already tested in this session
    asked_concepts = state.get("asked_concepts", [])
    avoid_hint = ""
    if asked_concepts:
        avoid_hint = (
            f"\nAlready-covered concepts (DO NOT ask about these again): "
            + ", ".join(f'"{c}"' for c in asked_concepts)
            + "\n"
        )

    questions_asked = state.get("questions_asked", 0)
    question_number = questions_asked + 1

    # ── Phase 5: Semantic Retrieval ──────────────────────────────────────────
    # Try to fetch focused chunks from the pre-embedded pgvector store instead
    # of using the entire day's notes (state['content']).
    notes_content = state['content']
    try:
        import src.memory.notion_memory as notion_memory
        
        # We need a query to search vectors. We use a broad prompt if this is
        # the first question, or we tell the search to find concepts NOT in
        # our asked_concepts list.
        search_query = f"core technical concepts excluding {', '.join(asked_concepts)}" if asked_concepts else "key technical concepts"
        
        # Search constrained to the date of this quiz session
        chunks = notion_memory.search_notes(
            query=search_query,
            k=4,
            date_filter=state['date'],
            score_threshold=0.85
        )
        
        if chunks:
            notes_content = "\n\n---\n\n".join(doc.page_content for doc in chunks)
            # We don't log here because print/log inside LangGraph nodes can be noisy,
            # but notion_memory itself logs the successful retrieval.
    except Exception as exc:
        # Graceful fallback to the legacy full-context approach
        pass

    system_prompt = f"""\
You are a quiz engine. Output ONLY ONE question — nothing else.

Rules (follow every single one):
1. The question must be about a specific technical fact, mechanism, or trade-off from the notes below.
2. Do NOT start with "Sure", "Great", "Of course", "Certainly", or any filler word.
3. Do NOT repeat or acknowledge any previous answer.
4. Do NOT include a preamble, a greeting, or any sentence other than the question itself.
5. The output must be a single interrogative sentence ending with a question mark.
6. Do NOT number the question or prefix it with "Question {question_number}:".
7. Choose a DIFFERENT topic/concept from any already covered — broaden coverage across the notes.{avoid_hint}
Notes from {state['date']}:
{notes_content}"""

    # Only pass the system message — no chat history so the LLM has nothing to "respond to"
    result: QuestionOutput = question_llm.invoke([SystemMessage(content=system_prompt)])

    return {
        "messages": [AIMessage(content=result.question)],
        "questions_asked": question_number,
        "is_completed": False
    }


def evaluate_answer(state: QuizState) -> Dict[str, Any]:
    """Evaluate the user's latest answer."""
    llm = get_llm(use_case="quiz", temperature=0)
    evaluator_llm = llm.with_structured_output(EvaluationOutput)
    
    system_prompt = f"""\
You are a strict technical evaluator. Your job is to judge whether the user answered the question correctly.

Rules:
- Set is_correct to True ONLY if the answer directly addresses the concept asked about AND demonstrates understanding.
- Set is_correct to False if the answer is off-topic, irrelevant, or simply wrong — even if it mentions a valid concept from elsewhere.
- Keep feedback to 1-2 sentences. State what was wrong or right. Do NOT use filler phrases like "Great try!" or "Correct concept".
- concept_tested should name the specific technical concept the question was probing.

Source material:
{state['content']}

Evaluate the user's most recent answer to the most recent question in the conversation."""
    messages = [SystemMessage(content=system_prompt)] + state.get("messages", [])[-2:]
    
    result: EvaluationOutput = evaluator_llm.invoke(messages)
    
    weak_areas = []
    if not result.is_correct:
        weak_areas.append(result.concept_tested)
    
    return {
        "evaluation_feedback": result.feedback,
        "last_concept": result.concept_tested,
        "last_confidence_score": result.confidence_score,
        "weak_areas": weak_areas,
        "asked_concepts": [result.concept_tested],  # accumulate every tested concept
    }


def route_next_step(state: QuizState) -> Literal["generate_question", "__end__"]:
    """Decide if we should ask another question or finish the quiz."""
    if state.get("questions_asked", 0) >= MAX_QUESTIONS:
        return "__end__"
    return "generate_question"


def build_quiz_graph(checkpointer: Optional[BaseCheckpointSaver] = None) -> StateGraph:
    """Compile and return the executable Quiz application.
    
    Args:
        checkpointer: The LangGraph state checkpointer instance.
    """
    workflow = StateGraph(QuizState)
    
    # ── Nodes ─────────────────────────────────────────────────────────────
    workflow.add_node("generate_question", generate_question)
    workflow.add_node("evaluate_answer", evaluate_answer)
    
    # ── Edges ─────────────────────────────────────────────────────────────
    workflow.add_edge(START, "generate_question")
    workflow.add_edge("generate_question", "evaluate_answer")
    
    # After evaluation, we conditionally route to generate another question or end.
    workflow.add_conditional_edges(
        "evaluate_answer",
        route_next_step,
        {
            "generate_question": "generate_question",
            "__end__": END
        }
    )
    
    # Compile with an interrupt before evaluate_answer
    # This means the graph will pause execution just before running evaluate_answer,
    # waiting for us to resume it (which we do when the user submits an answer).
    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["evaluate_answer"]
    )
