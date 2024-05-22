from typing_extensions import TypedDict
from typing import List


class GraphState(TypedDict):
    """|
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of retrieved documents
        all_documents: full context
        re_generate: rewrite the query question
        question_reasoning: store the explanation of context relevance reasoning
        answer_reasoning: store the explanation of generated answer relevance reasoning
        ans_regen: To call the next node based on answer relevance
    """

    question: str
    generation: str
    documents: List[str]
    all_documents: List[str]
    re_generate: str
    start_process: str
    answer_reasoning: str
    question_reasoning: str
    ans_regen: str
    retrieval_attempts: int


#
