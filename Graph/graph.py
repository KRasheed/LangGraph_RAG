from langgraph.graph import END, StateGraph
from Graph.States import GraphState
from Graph.Nodes_edges import (
    retrieve_agent,
    grade_documents,
    generate,
    transform_query,
    decide_to_generate,
    grade_generation_v_documents_and_question,
    Query_Assessment_Agent,
    decide_to_proceed,
    grade_generation,
)


def create_workflow():
    # graph_state = GraphState()
    workflow = StateGraph(GraphState)

    # add the edges
    workflow.add_node("Query_Assessment_Agent", Query_Assessment_Agent)
    workflow.add_node("retrieve_agent", retrieve_agent)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("grade_generation", grade_generation)

    # Build graph
    workflow.set_entry_point("Query_Assessment_Agent")
    workflow.add_conditional_edges(
        "Query_Assessment_Agent",
        decide_to_proceed,
        {
            "Continue": "retrieve_agent",
            "end": END,
        },
    )
    workflow.add_edge("retrieve_agent", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
            "No Context": END,
        },
    )

    workflow.add_edge("transform_query", "retrieve_agent")

    workflow.add_edge("generate", "grade_generation")

    workflow.add_conditional_edges(
        "grade_generation",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
        },
    )

    return workflow
