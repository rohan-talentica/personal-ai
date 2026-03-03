"""
RAG (Retrieval-Augmented Generation) chain with source citations.

Pattern from Day 3 & 4 notebooks — retrieves relevant docs from a Chroma
vector store then asks the LLM to answer using only that context, including
structured citations.
"""
from __future__ import annotations

from typing import Any

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough, Runnable
from operator import itemgetter

from src.utils.llm import get_llm, get_embeddings

_RAG_SYSTEM = """\
You are a precise AI assistant that answers questions strictly from the provided context.

Rules:
1. Base your answer ONLY on the context below — do not use prior knowledge.
2. At the end of every factual claim, add a citation in the format [Source: <title>].
3. If the context does not contain enough information, say "I don't have enough context to answer that."
4. Keep your answer concise and well-structured.

Context:
{context}
"""

_RAG_HUMAN = "Question: {question}"


def _format_docs(docs: list[Any]) -> str:
    """Convert a list of LangChain Documents into a numbered context block."""
    parts = []
    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get("title", doc.metadata.get("source", f"Document {i}"))
        parts.append(f"[{i}] Title: {title}\n{doc.page_content}")
    return "\n\n".join(parts)


def build_rag_chain(
    persist_directory: str,
    collection_name: str = "rag_collection",
    k: int = 4,
    model: str = "openai/gpt-3.5-turbo",
    temperature: float = 0.0,
) -> Runnable:
    """Build a RAG chain backed by a persisted ChromaDB collection.

    The returned chain expects a dict with:
        - ``question`` (str): the user's query

    Returns a dict with:
        - ``answer``  (str): LLM answer with inline citations
        - ``sources`` (list[dict]): list of retrieved document metadata

    Args:
        persist_directory: Path to the ChromaDB persist directory.
        collection_name: Name of the collection inside ChromaDB.
        k: Number of documents to retrieve per query.
        model: OpenRouter model identifier.
        temperature: Sampling temperature (0 recommended for RAG).
    """
    embeddings = get_embeddings()
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    llm = get_llm(model=model, temperature=temperature)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _RAG_SYSTEM),
            ("human", _RAG_HUMAN),
        ]
    )

    # Parallel branch: retrieve docs AND pass question through unchanged
    retrieve_and_pass = RunnableParallel(
        {
            "docs": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
    )

    def build_llm_input(inputs: dict) -> dict:
        return {
            "context": _format_docs(inputs["docs"]),
            "question": inputs["question"],
        }

    def build_output(inputs: dict) -> dict:
        return {
            "answer": inputs["answer"],
            "sources": [
                {
                    "title": d.metadata.get("title", d.metadata.get("source", "Unknown")),
                    "source": d.metadata.get("source", ""),
                    "document_type": d.metadata.get("document_type", ""),
                    "snippet": d.page_content[:200],
                }
                for d in inputs["docs"]
            ],
        }

    chain: Runnable = (
        retrieve_and_pass
        | RunnableLambda(
            lambda x: {
                **x,
                "answer": (prompt | llm | StrOutputParser()).invoke(build_llm_input(x)),
            }
        )
        | RunnableLambda(build_output)
    )
    return chain
