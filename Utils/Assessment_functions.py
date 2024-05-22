from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()
OPEN_API_KEY = os.environ["Openai_API_key"]


class GradeQuestion(BaseModel):
    """Binary score for relevance check on retrieved documents with explanation."""

    binary_score: str = Field(
        ...,
        description="Respond 'yes' if the question is relevant to the documents, otherwise 'no'.",
    )


def setup_query_grader(question, documents):
    # Initialize the LLM with function call
    llm = ChatOpenAI(
        model="gpt-3.5-turbo", temperature=0, api_key=OPEN_API_KEY
    )  # Adjusted model name for example
    structured_llm_grader = llm.with_structured_output(GradeQuestion)

    # Define the system prompt to include reasoning
    system_prompt = """
    "You are a relevance checker tasked with evaluating whether a user's question is relevant to the provided documents. 
    Review the context carefully and then assess the user's question. Respond with 'yes' if the question directly relates to the documents and 'no' if it does not, and explain your reasons. 
    
    """

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "User question: {question} \n\n documents: {documents}"),
        ]
    )

    # Combine the prompt template with the structured language model grader
    grader = grade_prompt | structured_llm_grader
    quest_grader = grader.invoke({"question": question, "documents": documents})
    return quest_grader


# Define the data model for grading documents
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents with explanation."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )
    explanation: str = Field(
        description="Explanation of why the document is or is not relevant to the question"
    )


def setup_retrieval_grader(Retrieved_document, question, document):
    # Initialize the LLM with function call
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, api_key=OPEN_API_KEY)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    # Define the system prompt to include reasoning
    system = """You are a grader tasked with evaluating the relevance of a retrieved document segment in response to a user's question. Determine whether this segment is the most pertinent excerpt from the document, effectively and precisely addressing the user's query.
    Grade the document as relevant ('yes') or not relevant ('no') and explain your detailed logical reasoning based on the presence of keyword(s) or semantic meanings related to the question."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Retrieved document: \n\n{Retrieved_document} \n\n User question: {question} \n\n User document: {document}",
            ),
        ]
    )

    # Combine the prompt template with the structured language model grader
    grader = grade_prompt | structured_llm_grader
    retrieval_grader = grader.invoke(
        {
            "question": question,
            "Retrieved_document": Retrieved_document,
            "document": document,
        }
    )
    return retrieval_grader


# Define the data model for evaluating answers
class EvaluateAnswers(BaseModel):
    """Model to store which answer is better and why."""

    best_answer: str = Field(
        description="Which answer is more correct, or specify if both are equally valid or invalid."
    )
    explanation: str = Field(
        description="Detailed explanation of why one answer is better, both are acceptable, or both are inadequate."
    )


def setup_answer_evaluator(context, question, answer1, answer2):
    """Initializes and returns an answer evaluator based on a structured output model and a detailed prompt."""
    # Initialize the language model with structured output
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, api_key=OPEN_API_KEY)
    structured_llm_evaluator = llm.with_structured_output(EvaluateAnswers)

    # System prompt to evaluate and compare answers
    system = """
    Assess the accuracy and relevancy of the two provided answers in relation to the given question. Use the context to determine:
    - If each answer directly addresses the question's topic.
    - If the answer is factually correct and relevant based on the context.

    Note:
    - Clearly state if neither answer addresses the question correctly or is factually incorrect.
    - Specify which answer, if any, is more relevant and why, even if neither is fully correct.
    """
    evaluate_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Context: {context} \n\n Question: {question} \n\n answer1: {answer1} \n\n answer2: {answer2}",
            ),
        ]
    )

    # Combine prompt template with structured output
    evaluator = evaluate_prompt | structured_llm_evaluator
    answer_evaluator = evaluator.invoke(
        {
            "context": context,
            "question": question,
            "answer1": answer1,
            "answer2": answer2,
        }
    )
    return answer_evaluator
