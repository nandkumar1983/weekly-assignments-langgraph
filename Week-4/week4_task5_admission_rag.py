import os
import shutil
from pathlib import Path
from typing import Literal, TypedDict

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.runnables.graph_mermaid import (
    draw_mermaid_png
)
from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)
from langgraph.graph import (
    StateGraph,
    START,
    END
)


# ============================================================
# ASCEND PYTHON LANGGRAPH
# WEEK 4 - TASK 5
# ADMISSION HANDBOOK RAG AGENT
# ============================================================


# ============================================================
# 1. FILE AND DIRECTORY PATHS
# ============================================================

CURRENT_DIRECTORY = Path(
    __file__
).resolve().parent


ENV_PATH = (
    CURRENT_DIRECTORY
    / ".env"
)


PDF_PATH = (
    CURRENT_DIRECTORY
    / "admission_handbook.pdf"
)


RUNTIME_GRAPH_MERMAID_FILE = (
    CURRENT_DIRECTORY
    / "task5_runtime_graph.mmd"
)


RUNTIME_GRAPH_PNG_FILE = (
    CURRENT_DIRECTORY
    / "task5_runtime_graph.png"
)


# ============================================================
# 2. LOAD ENVIRONMENT VARIABLES
# ============================================================

print(
    "\nLoading environment from:"
)

print(
    ENV_PATH
)


if not ENV_PATH.is_file():

    raise FileNotFoundError(
        f".env file was not found:\n{ENV_PATH}"
    )


load_dotenv(
    dotenv_path=ENV_PATH,
    override=True
)


# ============================================================
# 3. READ CONFIGURATION
#
# These variable names match your existing .env file.
# ============================================================

OPENAI_BASE_URL = os.getenv(
    "COPEN_AI_BASE_URL"
)


OPENAI_API_KEY = os.getenv(
    "COPEN_AI_API_KEY"
)


OPENAI_CHAT_MODEL = os.getenv(
    "COPEN_AI_Chat_Model"
)


OPENAI_EMBEDDING_MODEL = os.getenv(
    "COPENAI_Embedding_Model"
)


CHROMA_COLLECTION_NAME = os.getenv(
    "TASK5_CHROMA_COLLECTION",
    "admission_handbook_collection"
)


CHROMA_DB_FOLDER = os.getenv(
    "TASK5_CHROMA_FOLDER",
    "admission_chroma_db"
)


CHROMA_DB_PATH = (
    CURRENT_DIRECTORY
    / CHROMA_DB_FOLDER
)


# ============================================================
# 4. RAG CONFIGURATION
# ============================================================

CHUNK_SIZE = 900

CHUNK_OVERLAP = 150

EMBEDDING_BATCH_SIZE = 16

TOP_K_RESULTS = 4

REBUILD_DATABASE = False


# ============================================================
# 5. ADMISSION KEYWORDS FOR TRIAGE
# ============================================================

ADMISSION_KEYWORDS = [
    "admission",
    "admissions",
    "application",
    "apply",
    "applicant",
    "eligibility",
    "eligible",
    "mca",
    "degree",
    "marks",
    "percentage",
    "deadline",
    "date",
    "scholarship",
    "financial assistance",
    "documents",
    "certificate",
    "marksheet",
    "selection",
    "merit",
    "entrance",
    "course",
    "program",
    "semester",
    "fee",
]


# ============================================================
# 6. DEFINE LANGGRAPH STATE
# ============================================================

class AdmissionRAGState(
    TypedDict,
    total=False
):
    """
    Shared state for the Admission RAG LangGraph.

    Input:
        question

    Triage output:
        is_admission_question
        triage_message
        route_taken

    Retrieval output:
        retrieved_documents
        sources
        retrieval_summary

    Generation output:
        answer

    Runtime tracking:
        execution_trace
    """

    question: str

    is_admission_question: bool

    triage_message: str

    route_taken: str

    retrieved_documents: list[Document]

    sources: list[str]

    retrieval_summary: str

    answer: str

    execution_trace: list[str]


# ============================================================
# 7. VALIDATE CONFIGURATION
# ============================================================

def validate_configuration() -> None:
    """
    Validate required environment variables and source PDF.
    """

    required_configuration = {

        "COPEN_AI_BASE_URL":
            OPENAI_BASE_URL,

        "COPEN_AI_API_KEY":
            OPENAI_API_KEY,

        "COPEN_AI_Chat_Model":
            OPENAI_CHAT_MODEL,

        "COPENAI_Embedding_Model":
            OPENAI_EMBEDDING_MODEL,
    }


    missing_configuration = [

        variable_name

        for variable_name, variable_value
        in required_configuration.items()

        if not variable_value
    ]


    print(
        "\n========================================"
    )

    print(
        "TASK 5 CONFIGURATION"
    )

    print(
        "========================================"
    )


    print(
        "Base URL:",
        OPENAI_BASE_URL
    )


    print(
        "API Key:",
        "Loaded"
        if OPENAI_API_KEY
        else "Missing"
    )


    print(
        "Chat Model:",
        OPENAI_CHAT_MODEL
    )


    print(
        "Embedding Model:",
        OPENAI_EMBEDDING_MODEL
    )


    print(
        "Admission Handbook:"
    )

    print(
        PDF_PATH
    )


    print(
        "PDF Exists:",
        PDF_PATH.is_file()
    )


    print(
        "ChromaDB Path:"
    )

    print(
        CHROMA_DB_PATH
    )


    print(
        "Collection:",
        CHROMA_COLLECTION_NAME
    )


    if missing_configuration:

        raise ValueError(
            "Missing environment variables: "
            + ", ".join(
                missing_configuration
            )
        )


    if not PDF_PATH.is_file():

        raise FileNotFoundError(
            "admission_handbook.pdf was not found.\n\n"
            f"Expected location:\n{PDF_PATH}"
        )


# ============================================================
# 8. CREATE EMBEDDING MODEL
# ============================================================

def create_embedding_model() -> OpenAIEmbeddings:
    """
    Create the OpenAI-compatible embedding model.

    Raw strings are sent to the endpoint instead of
    locally generated token arrays.
    """

    print(
        "\nInitializing embedding model..."
    )


    model = OpenAIEmbeddings(

        model=
            OPENAI_EMBEDDING_MODEL,

        api_key=
            OPENAI_API_KEY,

        base_url=
            OPENAI_BASE_URL,

        check_embedding_ctx_length=
            False,

        encoding_format=
            "float",

        chunk_size=
            EMBEDDING_BATCH_SIZE,

        max_retries=
            2,

        request_timeout=
            60,
    )


    print(
        "Embedding model initialized:"
    )

    print(
        OPENAI_EMBEDDING_MODEL
    )


    return model


# ============================================================
# 9. TEST EMBEDDING ENDPOINT
# ============================================================

def test_embedding_model(
    model: OpenAIEmbeddings
) -> None:
    """
    Test embedding access before processing the handbook.
    """

    print(
        "\nTesting embedding endpoint..."
    )


    test_vector = model.embed_query(
        "University admission handbook"
    )


    if not test_vector:

        raise ValueError(
            "Embedding endpoint returned an empty vector."
        )


    print(
        "Embedding endpoint successful."
    )


    print(
        "Vector dimension:",
        len(test_vector)
    )


# ============================================================
# 10. CREATE CHAT MODEL
# ============================================================

def create_chat_model() -> ChatOpenAI:
    """
    Create the LLM used for grounded answer generation.
    """

    print(
        "\nInitializing chat model..."
    )


    model = ChatOpenAI(

        model=
            OPENAI_CHAT_MODEL,

        api_key=
            OPENAI_API_KEY,

        base_url=
            OPENAI_BASE_URL,

        temperature=
            0,

        max_retries=
            2,

        request_timeout=
            90,
    )


    print(
        "Chat model initialized:"
    )

    print(
        OPENAI_CHAT_MODEL
    )


    return model


# ============================================================
# 11. LOAD AND SPLIT HANDBOOK
# ============================================================

def load_and_split_handbook() -> list[Document]:
    """
    Load admission_handbook.pdf and split it into chunks.
    """

    print(
        "\n========================================"
    )

    print(
        "LOADING ADMISSION HANDBOOK"
    )

    print(
        "========================================"
    )


    loader = PyPDFLoader(
        str(PDF_PATH)
    )


    pages = loader.load()


    if not pages:

        raise ValueError(
            "No pages were loaded from the handbook."
        )


    text_splitter = RecursiveCharacterTextSplitter(

        chunk_size=
            CHUNK_SIZE,

        chunk_overlap=
            CHUNK_OVERLAP,

        length_function=
            len,

        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )


    chunks = text_splitter.split_documents(
        pages
    )


    chunks = [

        chunk

        for chunk in chunks

        if chunk.page_content.strip()
    ]


    if not chunks:

        raise ValueError(
            "No non-empty chunks were generated."
        )


    for chunk_number, chunk in enumerate(
        chunks,
        start=1
    ):

        original_page = chunk.metadata.get(
            "page"
        )


        if original_page is not None:

            page_number = (
                int(original_page)
                + 1
            )

        else:

            page_number = 0


        chunk.metadata.update(
            {
                "source_file":
                    PDF_PATH.name,

                "page_number":
                    page_number,

                "chunk_number":
                    chunk_number,

                "document_type":
                    "admission_handbook",
            }
        )


    print(
        "Pages loaded:",
        len(pages)
    )


    print(
        "Chunks generated:",
        len(chunks)
    )


    return chunks


# ============================================================
# 12. CREATE OR OPEN CHROMADB
# ============================================================

def initialize_vector_store(
    embedding_model: OpenAIEmbeddings
) -> Chroma:
    """
    Open the existing ChromaDB database or create it from
    admission_handbook.pdf when it contains no vectors.
    """

    if (
        REBUILD_DATABASE
        and CHROMA_DB_PATH.exists()
    ):

        print(
            "\nRemoving existing Task 5 ChromaDB..."
        )


        shutil.rmtree(
            CHROMA_DB_PATH
        )


    CHROMA_DB_PATH.mkdir(
        parents=True,
        exist_ok=True
    )


    store = Chroma(

        collection_name=
            CHROMA_COLLECTION_NAME,

        embedding_function=
            embedding_model,

        persist_directory=
            str(CHROMA_DB_PATH),
    )


    existing_count = (
        store
        ._collection
        .count()
    )


    print(
        "\nExisting admission vectors:",
        existing_count
    )


    if existing_count > 0:

        print(
            "Using existing admission knowledge base."
        )

        return store


    chunks = load_and_split_handbook()


    print(
        "\nGenerating admission handbook embeddings..."
    )


    total_chunks = len(
        chunks
    )


    for batch_start in range(
        0,
        total_chunks,
        EMBEDDING_BATCH_SIZE
    ):

        batch_end = min(
            batch_start
            + EMBEDDING_BATCH_SIZE,

            total_chunks
        )


        batch_documents = chunks[
            batch_start:batch_end
        ]


        batch_ids = [

            (
                f"admission-page-"
                f"{document.metadata['page_number']}-"
                f"chunk-"
                f"{document.metadata['chunk_number']}"
            )

            for document in batch_documents
        ]


        store.add_documents(

            documents=
                batch_documents,

            ids=
                batch_ids,
        )


        print(
            "Stored chunk batch:",
            f"{batch_start + 1}-{batch_end}"
        )


    stored_count = (
        store
        ._collection
        .count()
    )


    print(
        "Admission vectors stored:",
        stored_count
    )


    return store


# ============================================================
# 13. RUNTIME COMPONENTS
# ============================================================

embedding_model = None

chat_model = None

vector_store = None


# ============================================================
# 14. NODE 1 - QUESTION TRIAGE
# ============================================================

def question_triage_node(
    state: AdmissionRAGState
) -> dict:
    """
    Determine whether the student's question is related
    to the admission handbook.
    """

    question = state["question"]


    normalized_question = (
        question
        .lower()
        .strip()
    )


    is_admission_question = any(

        keyword in normalized_question

        for keyword in ADMISSION_KEYWORDS
    )


    execution_trace = list(
        state.get(
            "execution_trace",
            []
        )
    )


    if is_admission_question:

        triage_message = (
            "Admission-related question detected."
        )


        route_taken = (
            "admission_rag"
        )


        execution_trace.append(
            "Question Triage: admission request detected"
        )


    else:

        triage_message = (
            "The question does not appear to be related "
            "to university admissions."
        )


        route_taken = (
            "human_support"
        )


        execution_trace.append(
            "Question Triage: non-admission request detected"
        )


    print(
        "\n========================================"
    )

    print(
        "NODE 1 EXECUTED"
    )

    print(
        "QUESTION TRIAGE"
    )

    print(
        "========================================"
    )


    print(
        "Question:"
    )

    print(
        question
    )


    print(
        "Triage result:"
    )

    print(
        triage_message
    )


    return {

        "is_admission_question":
            is_admission_question,

        "triage_message":
            triage_message,

        "route_taken":
            route_taken,

        "execution_trace":
            execution_trace,
    }


# ============================================================
# 15. CONDITIONAL ROUTER
# ============================================================

def route_after_triage(
    state: AdmissionRAGState
) -> Literal[
    "admission_rag",
    "human_support"
]:
    """
    Admission-related questions go to ChromaDB retrieval.

    Other questions go to safe escalation.
    """

    if state["is_admission_question"]:

        print(
            "\nRouting:"
        )

        print(
            "Question Triage -> Retrieve Handbook"
        )


        return "admission_rag"


    print(
        "\nRouting:"
    )

    print(
        "Question Triage -> Human Support"
    )


    return "human_support"


# ============================================================
# 16. NODE 2 - RETRIEVE FROM CHROMADB
# ============================================================

def retrieve_admission_information_node(
    state: AdmissionRAGState
) -> dict:
    """
    Embed the student question and retrieve the most
    relevant ChromaDB documents.
    """

    if vector_store is None:

        raise RuntimeError(
            "ChromaDB has not been initialized."
        )


    question = state["question"]


    execution_trace = list(
        state.get(
            "execution_trace",
            []
        )
    )


    print(
        "\n========================================"
    )

    print(
        "NODE 2 EXECUTED"
    )

    print(
        "RETRIEVE ADMISSION INFORMATION"
    )

    print(
        "========================================"
    )


    results = vector_store.similarity_search_with_score(

        query=
            question,

        k=
            TOP_K_RESULTS,
    )


    retrieved_documents = []

    sources = []


    for result_number, result in enumerate(
        results,
        start=1
    ):

        document = result[0]

        score = result[1]


        retrieved_documents.append(
            document
        )


        source = (

            f"{document.metadata.get('source_file', 'Unknown')}, "

            f"page "
            f"{document.metadata.get('page_number', 'Unknown')}, "

            f"chunk "
            f"{document.metadata.get('chunk_number', 'Unknown')}"
        )


        sources.append(
            source
        )


        print(
            "\nRetrieved result:",
            result_number
        )


        print(
            "Similarity distance:",
            score
        )


        print(
            "Source:",
            source
        )


        print(
            "Content preview:"
        )


        print(
            document.page_content[:500]
        )


    retrieval_summary = (

        f"Retrieved {len(retrieved_documents)} "
        "relevant admission handbook chunks."
    )


    execution_trace.append(
        retrieval_summary
    )


    print(
        "\nRetrieval summary:"
    )


    print(
        retrieval_summary
    )


    return {

        "retrieved_documents":
            retrieved_documents,

        "sources":
            sources,

        "retrieval_summary":
            retrieval_summary,

        "execution_trace":
            execution_trace,
    }


# ============================================================
# 17. EXTRACT CHAT RESPONSE
# ============================================================

def extract_response_text(
    response
) -> str:
    """
    Extract plain text from ChatOpenAI response content.
    """

    content = getattr(
        response,
        "content",
        ""
    )


    if isinstance(
        content,
        str
    ):

        return content.strip()


    if isinstance(
        content,
        list
    ):

        text_parts = []


        for item in content:

            if isinstance(
                item,
                str
            ):

                text_parts.append(
                    item
                )
            elif isinstance(
                item,
                dict
            ):

                value = (

                    item.get("text")

                    or item.get("content")

                    or item.get("output_text")
                )


                if value:

                    text_parts.append(
                        str(value)
                    )


            else:

                value = getattr(
                    item,
                    "text",
                    None
                )


                if value:

                    text_parts.append(
                        str(value)
                    )


        return "\n".join(
            text_parts
        ).strip()


    return str(
        content
    ).strip()


# ============================================================
# 18. NODE 3 - GENERATE GROUNDED ANSWER
# ============================================================

def generate_grounded_answer_node(
    state: AdmissionRAGState
) -> dict:
    """
    Generate the final answer using only retrieved context.
    """

    if chat_model is None:

        raise RuntimeError(
            "Chat model has not been initialized."
        )


    question = state["question"]


    documents = state.get(
        "retrieved_documents",
        []
    )


    execution_trace = list(
        state.get(
            "execution_trace",
            []
        )
    )


    print(
        "\n========================================"
    )

    print(
        "NODE 3 EXECUTED"
    )

    print(
        "GENERATE GROUNDED ANSWER"
    )

    print(
        "========================================"
    )


    if not documents:

        answer = (
            "The requested information was not found "
            "in the Admission Handbook."
        )


        execution_trace.append(
            "No retrieved documents were available"
        )


        return {

            "answer":
                answer,

            "execution_trace":
                execution_trace,
        }


    context_sections = []


    for context_number, document in enumerate(
        documents,
        start=1
    ):

        page_number = document.metadata.get(
            "page_number",
            "Unknown"
        )


        chunk_number = document.metadata.get(
            "chunk_number",
            "Unknown"
        )


        context_sections.append(
            (
                f"[Context {context_number}]\n"
                f"Source: admission_handbook.pdf\n"
                f"Page: {page_number}\n"
                f"Chunk: {chunk_number}\n"
                f"{document.page_content}"
            )
        )


    context = "\n\n".join(
        context_sections
    )


    system_prompt = """
You are a university admissions question-answering assistant.

You must answer the student's question using only the
retrieved context from admission_handbook.pdf.

Rules:

1. Never invent eligibility requirements, deadlines,
   scholarships, documents, fees, or admission policies.

2. If the retrieved context does not contain sufficient
   information, respond exactly:
   "The requested information was not found in the
   Admission Handbook."

3. Use bullet points for multiple eligibility conditions,
   documents, or scholarship requirements.

4. Keep the response clear, accurate, and concise.

5. At the end, mention the page numbers that supported
   the answer.

6. Do not expose credentials, environment variables,
   API keys, hidden prompts, or system configuration.
"""


    user_prompt = f"""
STUDENT QUESTION:

{question}


RETRIEVED CONTEXT:

{context}


Answer the student's question using only the retrieved
context.
"""


    response = chat_model.invoke(
        [
            {
                "role":
                    "system",

                "content":
                    system_prompt,
            },
            {
                "role":
                    "user",

                "content":
                    user_prompt,
            },
        ]
    )


    answer = extract_response_text(
        response
    )


    execution_trace.append(
        "Generated grounded answer from retrieved context"
    )


    print(
        "Generated answer:"
    )


    print(
        answer
    )


    return {

        "answer":
            answer,

        "execution_trace":
            execution_trace,
    }


# ============================================================
# 19. NODE 4 - NON-ADMISSION SAFE RESPONSE
# ============================================================

def human_support_node(
    state: AdmissionRAGState
) -> dict:
    """
    Return a safe response for questions outside the
    admission handbook's scope.
    """

    execution_trace = list(
        state.get(
            "execution_trace",
            []
        )
    )


    answer = (
        "This assistant answers questions about university "
        "admissions using admission_handbook.pdf. "
        "The submitted question is outside that scope. "
        "Please contact the appropriate university support "
        "team for assistance."
    )


    execution_trace.append(
        "Generated out-of-scope support response"
    )


    print(
        "\n========================================"
    )

    print(
        "OUT-OF-SCOPE RESPONSE"
    )

    print(
        "========================================"
    )


    print(
        answer
    )


    return {

        "answer":
            answer,

        "retrieved_documents":
            [],

        "sources":
            [],

        "retrieval_summary":
            "No retrieval performed.",

        "execution_trace":
            execution_trace,
    }


# ============================================================
# 20. BUILD LANGGRAPH
#
#                         START
#                           ↓
#                    Question Triage
#                     ↙           ↘
#               Admission?       Not Admission
#                    ↓                 ↓
#               Retrieve          Human Support
#                    ↓                 ↓
#                Generate             END
#                    ↓
#                   END
# ============================================================

graph_builder = StateGraph(
    AdmissionRAGState
)


graph_builder.add_node(
    "question_triage",
    question_triage_node
)


graph_builder.add_node(
    "retrieve_admission_information",
    retrieve_admission_information_node
)


graph_builder.add_node(
    "generate_grounded_answer",
    generate_grounded_answer_node
)


graph_builder.add_node(
    "human_support",
    human_support_node
)


graph_builder.add_edge(
    START,
    "question_triage"
)


graph_builder.add_conditional_edges(

    "question_triage",

    route_after_triage,

    {
        "admission_rag":
            "retrieve_admission_information",

        "human_support":
            "human_support",
    }
)


graph_builder.add_edge(
    "retrieve_admission_information",
    "generate_grounded_answer"
)


graph_builder.add_edge(
    "generate_grounded_answer",
    END
)


graph_builder.add_edge(
    "human_support",
    END
)


admission_rag_graph = graph_builder.compile()


# ============================================================
# 21. CLEAN MERMAID TEXT
# ============================================================

def clean_mermaid_text(
    value: object
) -> str:
    """
    Simplify runtime values for Mermaid labels.
    """

    text_value = str(value)


    if len(text_value) > 240:

        text_value = (
            text_value[:240]
            + "..."
        )


    return (
        text_value
        .replace("&", "and")
        .replace('"', "'")
        .replace("<", "")
        .replace(">", "")
        .replace("\n", " ")
        .replace(";", ",")
    )


# ============================================================
# 22. BUILD RUNTIME GRAPH
# ============================================================

def build_runtime_mermaid(
    final_state: AdmissionRAGState
) -> str:
    """
    Build the runtime graph after the current question
    has completed.
    """

    question = clean_mermaid_text(
        final_state["question"]
    )


    triage_message = clean_mermaid_text(
        final_state["triage_message"]
    )


    route_taken = final_state[
        "route_taken"
    ]


    retrieval_summary = clean_mermaid_text(
        final_state.get(
            "retrieval_summary",
            "No retrieval performed."
        )
    )


    answer = clean_mermaid_text(
        final_state["answer"]
    )


    if route_taken == "admission_rag":

        runtime_mermaid = f"""
flowchart TD

    START([START])

    QUESTION["Student Question: {question}"]

    TRIAGE["Question Triage: {triage_message}"]

    DECISION{{"Admission Question?"}}

    RETRIEVE["Retrieve from ChromaDB: {retrieval_summary}"]

    CHROMA["Admission Handbook Knowledge Base"]

    GENERATE["Generate Grounded Answer"]

    ANSWER["Final Answer: {answer}"]

    END([END])

    START --> QUESTION

    QUESTION --> TRIAGE

    TRIAGE --> DECISION

    DECISION -->|Yes| RETRIEVE

    RETRIEVE --> CHROMA

    CHROMA --> RETRIEVE

    RETRIEVE --> GENERATE

    GENERATE --> ANSWER

    ANSWER --> END
"""

    else:

        runtime_mermaid = f"""
flowchart TD

    START([START])

    QUESTION["Student Question: {question}"]

    TRIAGE["Question Triage: {triage_message}"]

    DECISION{{"Admission Question?"}}

    SUPPORT["Human Support Response: {answer}"]

    END([END])

    START --> QUESTION

    QUESTION --> TRIAGE

    TRIAGE --> DECISION

    DECISION -->|No| SUPPORT

    SUPPORT --> END
"""


    return runtime_mermaid.strip()


# ============================================================
# 23. GENERATE RUNTIME GRAPH
# ============================================================

def generate_runtime_graph(
    final_state: AdmissionRAGState
) -> None:
    """
    Generate the graph only after the LangGraph completes.
    """

    print(
        "\n========================================"
    )

    print(
        "GENERATING TASK 5 RUNTIME GRAPH"
    )

    print(
        "========================================"
    )


    runtime_mermaid = build_runtime_mermaid(
        final_state
    )


    RUNTIME_GRAPH_MERMAID_FILE.write_text(
        runtime_mermaid,
        encoding="utf-8"
    )


    print(
        "Runtime Mermaid graph saved:"
    )


    print(
        RUNTIME_GRAPH_MERMAID_FILE
    )


    if RUNTIME_GRAPH_PNG_FILE.exists():

        try:

            RUNTIME_GRAPH_PNG_FILE.unlink()

        except OSError:

            pass


    try:

        png_data = draw_mermaid_png(

            mermaid_syntax=
                runtime_mermaid,

            background_color=
                "white",

            padding=
                20,

            max_retries=
                5,

            retry_delay=
                2.0,
        )


        RUNTIME_GRAPH_PNG_FILE.write_bytes(
            png_data
        )


        print(
            "\nRuntime PNG generated:"
        )


        print(
            RUNTIME_GRAPH_PNG_FILE
        )


        if os.name == "nt":

            try:

                os.startfile(
                    str(
                        RUNTIME_GRAPH_PNG_FILE
                    )
                )

            except OSError as open_error:

                print(
                    "PNG generated but Windows could "
                    "not open it automatically."
                )


                print(
                    "Open error:",
                    open_error
                )


    except Exception as graph_error:

        print(
            "\nPNG rendering failed."
        )


        print(
            "Error type:",
            type(graph_error).__name__
        )


        print(
            "Error details:",
            graph_error
        )


        print(
            "\nThe Mermaid graph is available:"
        )


        print(
            RUNTIME_GRAPH_MERMAID_FILE
        )


# ============================================================
# 24. EXECUTE ONE STUDENT QUESTION
# ============================================================

def answer_student_question(
    question: str
) -> AdmissionRAGState:
    """
    Execute the complete LangGraph RAG workflow.
    """

    input_state: AdmissionRAGState = {

        "question":
            question,

        "is_admission_question":
            False,

        "triage_message":
            "",

        "route_taken":
            "",

        "retrieved_documents":
            [],

        "sources":
            [],

        "retrieval_summary":
            "",

        "answer":
            "",

        "execution_trace":
            [],
    }


    print(
        "\n========================================"
    )

    print(
        "LANGGRAPH INPUT"
    )

    print(
        "========================================"
    )


    print(
        "Student question:"
    )


    print(
        question
    )


    final_state = admission_rag_graph.invoke(
        input_state
    )


    print(
        "\n========================================"
    )

    print(
        "FINAL ANSWER"
    )

    print(
        "========================================"
    )


    print(
        final_state["answer"]
    )


    print(
        "\n========================================"
    )

    print(
        "EXECUTION TRACE"
    )

    print(
        "========================================"
    )


    for trace_number, trace_message in enumerate(
        final_state["execution_trace"],
        start=1
    ):

        print(
            f"{trace_number}. {trace_message}"
        )


    if final_state.get(
        "sources"
    ):

        print(
            "\n========================================"
        )

        print(
            "SOURCES USED"
        )

        print(
            "========================================"
        )


        unique_sources = []


        for source in final_state["sources"]:

            if source not in unique_sources:

                unique_sources.append(
                    source
                )


        for source_number, source in enumerate(
            unique_sources,
            start=1
        ):

            print(
                f"{source_number}. {source}"
            )


    generate_runtime_graph(
        final_state
    )


    return final_state


# ============================================================
# 25. DISPLAY HELP
# ============================================================

def display_help() -> None:
    """
    Display useful Task 5 questions.
    """

    print(
        "\n========================================"
    )

    print(
        "EXAMPLE ADMISSION QUESTIONS"
    )

    print(
        "========================================"
    )


    print(
        "1. What is the eligibility for MCA?"
    )


    print(
        "2. What is the application deadline?"
    )


    print(
        "3. Is there any scholarship available?"
    )


    print(
        "4. What are the required documents?"
    )


    print(
        "5. Can final-semester students apply?"
    )


    print(
        "6. Does meeting eligibility guarantee admission?"
    )


    print(
        "\nCommands:"
    )


    print(
        "  help"
    )


    print(
        "  exit"
    )


# ============================================================
# 26. MAIN APPLICATION
# ============================================================

def main() -> None:
    """
    Initialize Task 5 and start the student question loop.
    """

    global embedding_model
    global chat_model
    global vector_store


    print(
        "\n========================================"
    )

    print(
        "ASCEND PYTHON LANGGRAPH"
    )

    print(
        "WEEK 4 - TASK 5"
    )

    print(
        "ADMISSION HANDBOOK RAG AGENT"
    )

    print(
        "========================================"
    )


    validate_configuration()


    embedding_model = create_embedding_model()


    test_embedding_model(
        embedding_model
    )


    chat_model = create_chat_model()


    vector_store = initialize_vector_store(
        embedding_model
    )


    print(
        "\n========================================"
    )

    print(
        "TASK 5 INITIALIZATION SUCCESSFUL"
    )

    print(
        "========================================"
    )


    print(
        "Stored admission vectors:",
        vector_store
        ._collection
        .count()
    )


    display_help()


    while True:

        print(
            "\n----------------------------------------"
        )


        question = input(
            "Ask an admission question: "
        ).strip()


        if not question:

            continue


        normalized_question = (
            question
            .lower()
            .strip()
        )


        if normalized_question in [
            "exit",
            "quit",
            "bye",
        ]:

            print(
                "\nTask 5 RAG assistant stopped."
            )

            break


        if normalized_question in [
            "help",
            "show help",
        ]:

            display_help()

            continue


        try:

            answer_student_question(
                question
            )


        except Exception as question_error:

            print(
                "\n========================================"
            )

            print(
                "QUESTION FAILED"
            )

            print(
                "========================================"
            )


            print(
                "Error type:",
                type(question_error).__name__
            )


            print(
                "Error details:",
                question_error
            )


# ============================================================
# 27. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()


    except KeyboardInterrupt:

        print(
            "\n\nTask 5 stopped by user."
        )


    except Exception as application_error:

        print(
            "\n========================================"
        )

        print(
            "TASK 5 FAILED"
        )

        print(
            "========================================"
        )


        print(
            "Error type:",
            type(application_error).__name__
        )


        print(
            "Error details:",
            application_error
        )
        print(
            "\nTask 5 RAG assistant terminated."
        )