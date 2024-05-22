from Utils.file_reader import setup_retriever
from Utils.Generation import setup_answer_generation as rag_chain
from Utils.Assessment_functions import setup_retrieval_grader as retrieval_grader
from Utils.Assessment_functions import setup_answer_evaluator as answer_evaluator
from Utils.Generation import setup_question_rewriter as question_rewriter
from Utils.Assessment_functions import setup_query_grader as query_evaluator
from dotenv import load_dotenv
import os
from Utils.cache import cache_manager

load_dotenv()
OPEN_API_KEY = os.environ["Openai_API_key"]


def Query_Assessment_Agent(state):
    print("---Assessment Agent---")
    question = state["question"]
    state["all_documents"] = cache_manager.get_data()
    all_docs = state[
        "all_documents"
    ]  # Assuming context is part of state now for relevance

    # print(f"Evaluating: {question} with Context: {all_docs}")
    query_grader = query_evaluator(question, all_docs)
    # print(f"Assessment Result: {query_grader['binary_score']}")

    if query_grader["binary_score"].lower() == "yes":
        print("Question is relevant, user can proceed.")
        state["start_process"] = "yes"
    else:
        state["start_process"] = "no"
        print("Question is not relevant.")
    return state


def retrieve_agent(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]
    state["all_documents"] = cache_manager.get_data()
    all_docs = state["all_documents"]
    # print('printing all docs')
    # print(all_docs)

    # print(f"State message: {state}")

    # Retrieval
    documents = setup_retriever(all_docs, question)
    return {"documents": documents, "question": question}


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    # RAG generation
    generation = rag_chain(documents, question)
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    all_docs = state["all_documents"]
    num_attempts = state[
        "retrieval_attempts"
    ]  # Ensure retrieval_attempts is properly initialized

    filtered_docs = []
    re_gen = "No"

    print("these are the number of attempts state value:", num_attempts)

    for d in documents:
        score = retrieval_grader(d.page_content, question, all_docs)
        grade = score["binary_score"]
        if grade == "yes":
            print("---The documents are relevant to the user's question---")
            print(f"Explanation: {score['explanation']}")
            filtered_docs.append(d)
            # print(filtered_docs)
        else:
            print("---The documents are not relevant to the user's question---")
            print(f"Explanation: {score['explanation']}")
            # print(f"Attempt {num_attempts} failed. No relevant documents found.")

    if not filtered_docs:
        re_gen = "Yes"
        if num_attempts is not None:
            num_attempts += 1  # Only increment if no documents were relevant
        else:
            num_attempts = 1  # Initialize to 1 if it was None
        # num_attempts += 1  # Only increment if no documents were relevant
        print(
            f"Attempt {num_attempts} failed. No relevant documents found. Will transform the query and retrieve again"
        )

    # Check if maximum attempts have been reached
    if num_attempts is not None and num_attempts >= 3 and not filtered_docs:
        re_gen = "No_context"
        print("Maximum attempts reached. I do not have the context of the question.")

    return {
        "documents": filtered_docs,
        "question": question,
        "re_generate": re_gen,
        "Assessment": score,
        "retrieval_attempts": num_attempts,
    }


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter(question)
    return {"documents": documents, "question": better_question}


def grade_generation(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Update the answer_regen state for decision to call next node.
    """

    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    second_ans = "This is a hypothetical answer"

    score = answer_evaluator(documents, question, generation, second_ans)
    relevance = score["best_answer"]

    # Determine which answer is more accurate or if neither is correct
    if relevance == "answer1":
        print(
            "---DECISION: GENERATION IS GROUNDED IN DOCUMENTS AND ANSWER 1 IS MORE RELEVANT---"
        )
        state["answer_reasoning"] = score["explanation"]
        state["ans_regen"] = "No"
        print(f"Explanation: {score['explanation']}")

    elif relevance == "answer2":
        print(
            "---DECISION: GENERATION IS GROUNDED IN DOCUMENTS AND ANSWER 2 IS MORE RELEVANT---"
        )
        state["answer_reasoning"] = score["explanation"]
        state["ans_regen"] = "No"
        print(f"Explanation: {score['explanation']}")

    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        state["ans_regen"] = "Yes"

    return {
        "ans_regen": state["ans_regen"],
        "answer_reasoning": state["answer_reasoning"],
    }


### Edges


def decide_to_proceed(state):
    """
    Determine whether to start the retrieval procedure or should end to start again.
      Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    start_process = state["start_process"]
    if start_process == "yes":
        return "Continue"
    else:
        return "end"


def decide_to_retrieve(state):
    """
    Determines whether to retrieve the documents again based on transformed query.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    if "transform_query":
        print("---DECISION: Retrieving documents again based on transformed query---")
        return "retrieve"


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    # print("---Checking the relevance of query: ---")
    re_gen = state["re_generate"]

    if re_gen == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    elif re_gen == "No_context":
        print("I do not have the context of the question.")
        return "No Context"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


def grade_generation_v_documents_and_question(state):
    """
    Conditional edge to move to the next state. If answer is not grounded regenerate the answer else end
    the process

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    if state["ans_regen"] == "No":
        return "useful"
    else:
        return "not supported"
