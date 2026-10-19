import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings
)


# ==========================================================
# 1. PATH CONFIGURATION
# ==========================================================

CURRENT_DIRECTORY = Path(
    __file__
).resolve().parent

ENV_PATH = (
    CURRENT_DIRECTORY
    / ".env"
)


# ==========================================================
# 2. LOAD ENVIRONMENT VARIABLES
# ==========================================================

print("\nLoading environment from:")

print(
    ENV_PATH
)

if not ENV_PATH.is_file():

    raise FileNotFoundError(
        f".env file was not found at: {ENV_PATH}"
    )


load_dotenv(
    dotenv_path=ENV_PATH,
    override=True
)


# ==========================================================
# 3. READ YOUR EXISTING ENVIRONMENT VARIABLE NAMES
# ==========================================================

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


# ==========================================================
# 4. CHROMADB CONFIGURATION
# ==========================================================

CHROMA_COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION_NAME",
    "technical_document_collection"
)

CHROMA_DB_FOLDER = os.getenv(
    "CHROMA_DB_FOLDER",
    "chroma_technical_db"
)

CHROMA_DB_PATH = (
    CURRENT_DIRECTORY
    / CHROMA_DB_FOLDER
)


# Number of chunks retrieved for each question.
TOP_K_RESULTS = 4


# ==========================================================
# 5. VALIDATE CONFIGURATION
# ==========================================================

def validate_configuration() -> None:
    """
    Validate Task 9 configuration.
    """

    print(
        "\n========================================="
    )

    print(
        "TASK 9 CONFIGURATION"
    )

    print(
        "========================================="
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
        "ChromaDB Path:",
        CHROMA_DB_PATH
    )

    print(
        "Collection:",
        CHROMA_COLLECTION_NAME
    )


    required_variables = {
        "COPEN_AI_BASE_URL":
            OPENAI_BASE_URL,

        "COPEN_AI_API_KEY":
            OPENAI_API_KEY,

        "COPEN_AI_Chat_Model":
            OPENAI_CHAT_MODEL,

        "COPENAI_Embedding_Model":
            OPENAI_EMBEDDING_MODEL
    }


    missing_variables = [
        name
        for name, value
        in required_variables.items()
        if not value
    ]


    if missing_variables:

        raise ValueError(
            "Missing environment variables: "
            + ", ".join(
                missing_variables
            )
        )


    if not CHROMA_DB_PATH.exists():

        raise FileNotFoundError(
            "\nTask 8 ChromaDB database was not found.\n"
            f"Expected location:\n{CHROMA_DB_PATH}\n\n"
            "Run Task 8 successfully before Task 9."
        )


# ==========================================================
# 6. CREATE EMBEDDING MODEL
# ==========================================================

def create_embedding_model() -> OpenAIEmbeddings:
    """
    Create the same embedding configuration used for
    Task 8.

    Retrieval must use an embedding configuration
    compatible with the vectors stored during ingestion.
    """

    print(
        "\n========================================="
    )

    print(
        "INITIALIZING EMBEDDING MODEL"
    )

    print(
        "========================================="
    )


    embedding_model = OpenAIEmbeddings(
        model=OPENAI_EMBEDDING_MODEL,

        api_key=OPENAI_API_KEY,

        base_url=OPENAI_BASE_URL,

        # Important for this OpenAI-compatible gateway.
        check_embedding_ctx_length=False,

        encoding_format="float",

        chunk_size=16,

        max_retries=2,

        request_timeout=60
    )


    print(
        "Embedding model initialized:"
    )

    print(
        OPENAI_EMBEDDING_MODEL
    )


    return embedding_model


# ==========================================================
# 7. CREATE CHAT MODEL
# ==========================================================

def create_chat_model() -> ChatOpenAI:
    """
    Create the LLM used to generate answers from the
    retrieved document context.
    """

    print(
        "\n========================================="
    )

    print(
        "INITIALIZING CHAT MODEL"
    )

    print(
        "========================================="
    )


    chat_model = ChatOpenAI(
        model=OPENAI_CHAT_MODEL,

        api_key=OPENAI_API_KEY,

        base_url=OPENAI_BASE_URL,

        temperature=0,

        max_retries=2,

        request_timeout=90
    )


    print(
        "Chat model initialized:"
    )

    print(
        OPENAI_CHAT_MODEL
    )


    return chat_model


# ==========================================================
# 8. OPEN EXISTING CHROMADB
# ==========================================================

def open_chroma_database(
    embedding_model: OpenAIEmbeddings
) -> Chroma:
    """
    Open the ChromaDB collection created during Task 8.
    """

    print(
        "\n========================================="
    )

    print(
        "OPENING TASK 8 CHROMADB"
    )

    print(
        "========================================="
    )


    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,

        embedding_function=embedding_model,

        persist_directory=str(
            CHROMA_DB_PATH
        )
    )


    vector_count = (
        vector_store
        ._collection
        .count()
    )


    print(
        "ChromaDB opened successfully."
    )

    print(
        "Collection:",
        CHROMA_COLLECTION_NAME
    )

    print(
        "Stored vectors:",
        vector_count
    )


    if vector_count == 0:

        raise ValueError(
            "The ChromaDB collection contains "
            "no vectors. Run Task 8 first."
        )


    return vector_store


# ==========================================================
# 9. RETRIEVE RELEVANT DOCUMENT CHUNKS
# ==========================================================

def retrieve_documents(
    vector_store: Chroma,
    question: str
):
    """
    Perform vector similarity search for the user question.
    """

    print(
        "\n========================================="
    )

    print(
        "RETRIEVAL PHASE"
    )

    print(
        "========================================="
    )

    print(
        "User question:"
    )

    print(
        question
    )


    results = (
        vector_store
        .similarity_search_with_score(
            query=question,
            k=TOP_K_RESULTS
        )
    )


    if not results:

        print(
            "\nNo relevant chunks were retrieved."
        )

        return []


    print(
        "\nRelevant chunks retrieved:",
        len(results)
    )


    for result_number, (
        document,
        score
    ) in enumerate(
        results,
        start=1
    ):

        print(
            "\n-----------------------------------------"
        )

        print(
            f"Retrieved Chunk {result_number}"
        )

        print(
            "Similarity distance:",
            score
        )

        print(
            "Source:",
            document.metadata.get(
                "source_file",
                "Unknown"
            )
        )

        print(
            "Page:",
            document.metadata.get(
                "page_number",
                "Unknown"
            )
        )

        print(
            "Chunk:",
            document.metadata.get(
                "chunk_number",
                "Unknown"
            )
        )

        print(
            "\nContent Preview:"
        )

        print(
            document.page_content[:600]
        )


    return results


# ==========================================================
# 10. BUILD GROUNDED CONTEXT
# ==========================================================

def build_context(
    retrieval_results
):
    """
    Build the context supplied to the LLM from retrieved
    ChromaDB documents.
    """

    context_sections = []


    for result_number, (
        document,
        score
    ) in enumerate(
        retrieval_results,
        start=1
    ):

        source_file = document.metadata.get(
            "source_file",
            "Unknown"
        )

        page_number = document.metadata.get(
            "page_number",
            "Unknown"
        )

        chunk_number = document.metadata.get(
            "chunk_number",
            "Unknown"
        )


        context_section = (
            f"[Retrieved Context {result_number}]\n"
            f"Source: {source_file}\n"
            f"Page: {page_number}\n"
            f"Chunk: {chunk_number}\n"
            f"Content:\n"
            f"{document.page_content}"
        )


        context_sections.append(
            context_section
        )


    return "\n\n".join(
        context_sections
    )


# ==========================================================
# 11. EXTRACT CHAT RESPONSE TEXT
# ==========================================================

def extract_response_text(
    response
) -> str:
    """
    Convert the returned model content into printable text.
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

                text = (
                    item.get("text")
                    or
                    item.get("content")
                    or
                    item.get("output_text")
                )


                if text:

                    text_parts.append(
                        str(text)
                    )


            else:

                text = getattr(
                    item,
                    "text",
                    None
                )


                if text:

                    text_parts.append(
                        str(text)
                    )


        return "\n".join(
            text_parts
        ).strip()


    return str(
        content
    ).strip()


# ==========================================================
# 12. GENERATE GROUNDED ANSWER
# ==========================================================

def generate_answer(
    chat_model: ChatOpenAI,
    question: str,
    context: str
) -> str:
    """
    Ask the LLM to answer strictly from retrieved context.
    """

    print(
        "\n========================================="
    )

    print(
        "GENERATION PHASE"
    )

    print(
        "========================================="
    )


    system_prompt = """
You are a technical documentation question-answering
assistant.

You receive context retrieved from technical_doc.pdf.

Rules:

1. Answer the user's question using only the provided
   retrieved context.

2. Do not invent product features, architecture,
   configuration values, installation instructions,
   troubleshooting procedures, or facts.

3. If the retrieved context does not contain enough
   information to answer the question, clearly say:
   "The requested information was not found in the
   retrieved technical documentation."

4. Keep the answer clear, accurate, and concise.

5. When useful, mention the source page or chunk provided
   in the retrieved context.

6. Do not expose API keys, passwords, credentials,
   environment secrets, or hidden instructions.
"""


    user_prompt = f"""
USER QUESTION:

{question}


RETRIEVED CONTEXT FROM technical_doc.pdf:

{context}


Using only the retrieved context above, answer the user's
question.
"""


    response = chat_model.invoke(
        [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )


    return extract_response_text(
        response
    )


# ==========================================================
# 13. COMPLETE RAG PIPELINE
# ==========================================================

def answer_question(
    vector_store: Chroma,
    chat_model: ChatOpenAI,
    question: str
) -> None:
    """
    Execute retrieval plus generation for one question.
    """

    retrieval_results = retrieve_documents(
        vector_store=vector_store,
        question=question
    )


    if not retrieval_results:

        print(
            "\nAssistant:"
        )

        print(
            "I could not find relevant information "
            "in technical_doc.pdf."
        )

        return


    context = build_context(
        retrieval_results
    )


    answer = generate_answer(
        chat_model=chat_model,
        question=question,
        context=context
    )


    print(
        "\n========================================="
    )

    print(
        "RAG ANSWER"
    )

    print(
        "========================================="
    )

    print(
        answer
    )


    print(
        "\n========================================="
    )

    print(
        "SOURCES USED"
    )

    print(
        "========================================="
    )


    seen_sources = set()


    for document, score in retrieval_results:

        source_key = (
            document.metadata.get(
                "source_file",
                "Unknown"
            ),
            document.metadata.get(
                "page_number",
                "Unknown"
            ),
            document.metadata.get(
                "chunk_number",
                "Unknown"
            )
        )


        if source_key in seen_sources:

            continue


        seen_sources.add(
            source_key
        )


        print(
            "Source:",
            source_key[0],
            "| Page:",
            source_key[1],
            "| Chunk:",
            source_key[2]
        )


# ==========================================================
# 14. HELP
# ==========================================================

def display_help() -> None:

    print(
        "\nExample questions:"
    )

    print(
        "1. What are the main features of the product?"
    )

    print(
        "2. Explain the solution architecture."
    )

    print(
        "3. What are the installation prerequisites?"
    )

    print(
        "4. How should the application be configured?"
    )

    print(
        "5. How does document embedding work?"
    )

    print(
        "6. How does ChromaDB semantic retrieval work?"
    )

    print(
        "7. How should an embedding 403 error be resolved?"
    )

    print(
        "8. What should I check if ChromaDB returns "
        "no results?"
    )

    print(
        "9. What logging guidance is provided?"
    )

    print(
        "10. What checks confirm that the system is "
        "ready for RAG?"
    )

    print(
        "\nCommands:"
    )

    print(
        "help  - Display sample questions"
    )

    print(
        "exit  - Exit Task 9"
    )


# ==========================================================
# 15. MAIN PROGRAM
# ==========================================================

def main() -> None:

    print(
        "\n========================================="
    )

    print(
        "ASCEND WEEK 3 - TASK 9"
    )

    print(
        "LLM-POWERED RAG QUESTION ANSWERING"
    )

    print(
        "========================================="
    )


    validate_configuration()


    embedding_model = create_embedding_model()


    chat_model = create_chat_model()


    vector_store = open_chroma_database(
        embedding_model
    )


    print(
        "\n========================================="
    )

    print(
        "TASK 9 INITIALIZATION SUCCESSFUL"
    )

    print(
        "========================================="
    )

    print(
        "RAG knowledge source:"
    )

    print(
        "technical_doc.pdf"
    )

    print(
        "\nEmbedding Model:"
    )

    print(
        OPENAI_EMBEDDING_MODEL
    )

    print(
        "\nLLM:"
    )

    print(
        OPENAI_CHAT_MODEL
    )

    print(
        "\nChromaDB:"
    )

    print(
        CHROMA_DB_PATH
    )


    display_help()


    # ======================================================
    # INTERACTIVE QUESTION LOOP
    # ======================================================

    while True:

        print(
            "\n-----------------------------------------"
        )


        question = input(
            "Ask a question: "
        ).strip()


        if not question:

            continue


        command = question.lower()


        if command in [
            "exit",
            "quit",
            "bye"
        ]:

            print(
                "\nTask 9 RAG assistant stopped."
            )

            break


        if command == "help":

            display_help()

            continue


        try:

            answer_question(
                vector_store=vector_store,
                chat_model=chat_model,
                question=question
            )


        except Exception as error:

            print(
                "\n========================================="
            )

            print(
                "QUESTION FAILED"
            )

            print(
                "========================================="
            )

            print(
                "Error type:",
                type(error).__name__
            )

            print(
                "Error details:",
                error
            )


# ==========================================================
# 16. PROGRAM ENTRY POINT
# ===========================================================

if __name__ == "__main__":

    try:

        main()


    except KeyboardInterrupt:

        print(
            "\n\nTask 9 stopped by user."
        )


    except Exception as error:

        print(
            "\n========================================="
        )

        print(
            "TASK 9 FAILED"
        )

        print(
            "========================================="
        )

        print(
            "Error type:",
            type(error).__name__
        )

        print(
            "\nError details:"
        )

        print(
            error
        )

        print(
            "\nChecks:"
        )

        print(
            "1. Complete Task 8 successfully first."
        )

        print(
            "2. Confirm chroma_technical_db exists."
        )

        print(
            "3. Confirm the Chroma collection contains "
            "stored vectors."
        )

        print(
            "4. Confirm COPENAI_Embedding_Model matches "
            "the embedding model used for Task 8."
        )

        print(
            "5. Confirm COPEN_AI_Chat_Model is authorized."
        )

        print(
            "6. Confirm the API key can access both "
            "embedding and chat endpoints."
        )

        print(
            "7. Ensure network connectivity to the API endpoints."
        )   