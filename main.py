from Graph.graph import create_workflow
from Utils.file_reader import get_doc_splits
from dotenv import load_dotenv
import os

from Utils.cache import cache_manager

load_dotenv()
openai_api_key = os.environ["Openai_API_key"]

if __name__ == "__main__":
    data = get_doc_splits("C:/Users/LTC/Projects/Retreival testing/Data")

    cache_manager.set_data(data)  # Storing the

    # cached_documents = cache_manager.get_data()
    # print(cached_documents)
    # Now initialize and start your workflow
    # graph_state = initialize_graph_state()
    # print(graph_state)

    work_flow = create_workflow()
    app = work_flow.compile()

    question = input("question: ")
    question_dic = {"question": question}
    print(question_dic)
    # {"question": "What were the key discoveries made in the 1980s that advanced our understanding of the causes of Duchenne Muscular Dystrophy (DMD)?"}
    # for output in app.stream(inputs):
    #     for key, value in output.items():
    #         print(f"Output from node '{key}':")
    #         print("---")
    #         print(value)
    #         print("\n---\n")
    answer = app.invoke(question_dic)
    print(answer["generation"])
