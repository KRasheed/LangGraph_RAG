from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
OPEN_API_KEY = os.environ["Openai_API_key"]


def setup_question_rewriter(question):
    """Initializes and returns a question re-writer configured with a structured output model and a detailed prompt."""
    # Initialize the LLM with specific settings
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, api_key=OPEN_API_KEY)

    # Define the system prompt for rewriting questions
    system = """You are a question re-writer that converts an input question to a better version that is optimized
for web search. Look at the input and try to reason about the underlying semantic intent/meaning."""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )

    # Combine the prompt template with the LLM and parse the output to string
    rewriter = re_write_prompt | llm | StrOutputParser()
    question_rewriter = rewriter.invoke({"question": question})
    return question_rewriter


def setup_answer_generation(context, question):
    # Prompt
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    # LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=OPEN_API_KEY)

    # Chain
    chain = prompt | llm | StrOutputParser()
    rag_chain = chain.invoke({"context": context, "question": question})

    return rag_chain
