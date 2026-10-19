import hashlib
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==========================================================
# 1. FILE AND DIRECTORY PATHS
# ==========================================================

CURRENT_DIRECTORY = Path(__file__).resolve().parent

ENV_PATH = CURRENT_DIRECTORY / ".env"

PDF_PATH = CURRENT_DIRECTORY / "technical_doc.pdf"


# ==========================================================
# 2. LOAD ENVIRONMENT VARIABLES
# ==========================================================

print("\nLoading environment from:")
print(ENV_PATH)

if not ENV_PATH.is_file():
    raise FileNotFoundError(
        f".env file was not found at: {ENV_PATH}"
    )

load_dotenv(
    dotenv_path=ENV_PATH,
    override=True
)


# ==========================================================
# 3. READ CONFIGURATION
#
# These names exactly match the names in your .env file.
# Environment-variable names are case-sensitive.
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
# 4. CHROMADB SETTINGS
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


# ==========================================================
# 5. INGESTION SETTINGS
# ==========================================================

CHUNK_SIZE = 1000

CHUNK_OVERLAP = 200

EMBEDDING_BATCH_SIZE = 16

SEARCH_RESULT_COUNT = 3

RESET_DATABASE_BEFORE_INGESTION = True


# ==========================================================
# 6. VALIDATE CONFIGURATION
# ==========================================================

def validate_configuration() -> None:
    """
    Validate environment variables and required input files.
    """

    required_configuration = {
        "COPEN_AI_BASE_URL":
            OPENAI_BASE_URL,

        "COPEN_AI_API_KEY":
            OPENAI_API_KEY,

        "COPEN_AI_Chat_Model":
            OPENAI_CHAT_MODEL,

        "COPENAI_Embedding_Model":
            OPENAI_EMBEDDING_MODEL
    }

    missing_configuration = [
        variable_name
        for variable_name, variable_value
        in required_configuration.items()
        if not variable_value
    ]

    print("\n========== CONFIGURATION CHECK ==========")

    print(
        "Base URL              :",
        OPENAI_BASE_URL
    )

    print(
        "API Key               :",
        "Loaded" if OPENAI_API_KEY else "Missing"
    )

    print(
        "Chat Model            :",
        OPENAI_CHAT_MODEL
    )

    print(
        "Embedding Model       :",
        OPENAI_EMBEDDING_MODEL
    )

    print(
        "PDF Path              :",
        PDF_PATH
    )

    print(
        "PDF Exists            :",
        PDF_PATH.is_file()
    )

    print(
        "ChromaDB Path         :",
        CHROMA_DB_PATH
    )

    print(
        "Collection Name       :",
        CHROMA_COLLECTION_NAME
    )

    print(
        "Chunk Size            :",
        CHUNK_SIZE
    )

    print(
        "Chunk Overlap         :",
        CHUNK_OVERLAP
    )

    print(
        "Embedding Batch Size  :",
        EMBEDDING_BATCH_SIZE
    )

    print("=========================================")

    if missing_configuration:
        raise ValueError(
            "Missing environment variables: "
            + ", ".join(missing_configuration)
        )

    if not PDF_PATH.is_file():
        raise FileNotFoundError(
            "technical_doc.pdf was not found.\n"
            f"Expected location: {PDF_PATH}"
        )

    if OPENAI_EMBEDDING_MODEL == OPENAI_CHAT_MODEL:
        raise ValueError(
            "The chat model and embedding model cannot "
            "be the same.\n\n"
            f"Chat model: {OPENAI_CHAT_MODEL}\n"
            f"Embedding model: {OPENAI_EMBEDDING_MODEL}"
        )


# ==========================================================
# 7. LOAD PDF
# ==========================================================

def load_pdf_document(
    pdf_path: Path
) -> list[Document]:
    """
    Load the PDF as page-level LangChain documents.
    """

    print("\n=========================================")
    print("STEP 1: LOADING PDF")
    print("=========================================")

    loader = PyPDFLoader(
        str(pdf_path)
    )

    page_documents = loader.load()

    if not page_documents:
        raise ValueError(
            "The PDF loader returned no pages."
        )

    total_characters = sum(
        len(document.page_content)
        for document in page_documents
    )

    non_empty_pages = sum(
        1
        for document in page_documents
        if document.page_content.strip()
    )

    print("PDF loaded successfully.")

    print(
        "Total pages loaded:",
        len(page_documents)
    )

    print(
        "Pages containing text:",
        non_empty_pages
    )

    print(
        "Total extracted characters:",
        total_characters
    )

    if total_characters == 0:
        raise ValueError(
            "The PDF contains no extractable text. "
            "The PDF may require OCR."
        )

    return page_documents


# ==========================================================
# 8. SPLIT PDF INTO CHUNKS
# ==========================================================

def split_documents(
    page_documents: list[Document]
) -> list[Document]:
    """
    Split PDF pages into manageable overlapping chunks.
    """

    print("\n=========================================")
    print("STEP 2: SPLITTING DOCUMENT")
    print("=========================================")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    document_chunks = (
        text_splitter.split_documents(
            page_documents
        )
    )

    document_chunks = [
        chunk
        for chunk in document_chunks
        if chunk.page_content.strip()
    ]

    if not document_chunks:
        raise ValueError(
            "No non-empty chunks were created."
        )

    for chunk_number, chunk in enumerate(
        document_chunks,
        start=1
    ):
        original_page = chunk.metadata.get(
            "page"
        )

        if original_page is not None:
            page_number = int(original_page) + 1
        else:
            page_number = 0

        chunk.metadata.update(
            {
                "source_file":
                    PDF_PATH.name,

                "source_path":
                    str(PDF_PATH),

                "page_number":
                    page_number,

                "chunk_number":
                    chunk_number,

                "character_count":
                    len(chunk.page_content),

                "collection_name":
                    CHROMA_COLLECTION_NAME
            }
        )

    chunk_lengths = [
        len(chunk.page_content)
        for chunk in document_chunks
    ]

    print(
        "Chunks created:",
        len(document_chunks)
    )

    print(
        "Smallest chunk:",
        min(chunk_lengths),
        "characters"
    )

    print(
        "Largest chunk:",
        max(chunk_lengths),
        "characters"
    )

    print(
        "Average chunk size:",
        round(
            sum(chunk_lengths)
            / len(chunk_lengths),
            2
        ),
        "characters"
    )

    print("\nFirst chunk preview:")

    print(
        document_chunks[0]
        .page_content[:700]
    )

    return document_chunks


# ==========================================================
# 9. GENERATE STABLE CHUNK IDS
# ==========================================================

def generate_chunk_ids(
    document_chunks: list[Document]
) -> list[str]:
    """
    Generate deterministic SHA-256 IDs for every chunk.
    """

    chunk_ids = []

    for chunk in document_chunks:
        identifier_source = (
            f"{chunk.metadata.get('source_file')}|"
            f"{chunk.metadata.get('page_number')}|"
            f"{chunk.metadata.get('chunk_number')}|"
            f"{chunk.page_content}"
        )

        chunk_hash = hashlib.sha256(
            identifier_source.encode("utf-8")
        ).hexdigest()

        chunk_ids.append(
            chunk_hash
        )

    print(
        "\nStable chunk IDs generated:",
        len(chunk_ids)
    )

    return chunk_ids


# ==========================================================
# 10. INITIALIZE EMBEDDING MODEL
# ==========================================================

def create_embedding_model() -> OpenAIEmbeddings:
    """
    Initialize the OpenAI-compatible embedding client.

    check_embedding_ctx_length=False sends raw text strings
    instead of locally generated integer token arrays.
    """

    print("\n=========================================")
    print("STEP 3: INITIALIZING EMBEDDING MODEL")
    print("=========================================")

    embedding_model = OpenAIEmbeddings(
        model=OPENAI_EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,

        # Important for this OpenAI-compatible gateway.
        check_embedding_ctx_length=False,

        # Request numeric floating-point vectors.
        encoding_format="float",

        # Keep API batches controlled.
        chunk_size=EMBEDDING_BATCH_SIZE,

        max_retries=2,
        request_timeout=60
    )

    print("Embedding model initialized:")

    print(
        OPENAI_EMBEDDING_MODEL
    )

    print(
        "Raw-string input enabled:",
        True
    )

    print(
        "Encoding format:",
        "float"
    )

    return embedding_model


# ==========================================================
# 11. TEST EMBEDDING ENDPOINT
# ==========================================================

def test_embedding_model(
    embedding_model: OpenAIEmbeddings
) -> int:
    """
    Generate a single test vector before PDF ingestion.
    """

    print("\nTesting embedding endpoint...")

    test_text = (
        "Technical documentation embedding test"
    )

    print(
        "Test input type:",
        type(test_text).__name__
    )

    print(
        "Test input value:"
    )

    print(
        test_text
    )

    test_vector = embedding_model.embed_query(
        test_text
    )

    if not test_vector:
        raise ValueError(
            "The embedding endpoint returned "
            "an empty vector."
        )

    if not isinstance(test_vector, list):
        raise TypeError(
            "The embedding endpoint did not return "
            "a Python list."
        )

    vector_dimension = len(
        test_vector
    )

    print(
        "\nEmbedding endpoint: Successful"
    )

    print(
        "Vector type:",
        type(test_vector).__name__
    )

    print(
        "Vector dimension:",
        vector_dimension
    )

    print(
        "First five vector values:"
    )

    print(
        test_vector[:5]
    )
    
    return vector_dimension


# ==========================================================
# 12. RESET EXISTING CHROMADB
# ==========================================================

def reset_chroma_database() -> None:
    """
    Delete the previous local ChromaDB database.
    """

    if not RESET_DATABASE_BEFORE_INGESTION:
        print(
            "\nExisting ChromaDB will be retained."
        )

        return

    if not CHROMA_DB_PATH.exists():
        print(
            "\nNo previous ChromaDB directory exists."
        )

        return

    print(
        "\nRemoving previous ChromaDB directory:"
    )

    print(
        CHROMA_DB_PATH
    )

    shutil.rmtree(
        CHROMA_DB_PATH
    )

    print(
        "Previous ChromaDB directory removed."
    )


# ==========================================================
# 13. CREATE CHROMADB VECTOR STORE
# ==========================================================

def create_chroma_vector_store(
    document_chunks: list[Document],
    chunk_ids: list[str],
    embedding_model: OpenAIEmbeddings
) -> Chroma:
    """
    Generate embeddings and store vectors in ChromaDB.
    """

    print("\n=========================================")
    print("STEP 4: GENERATING AND STORING EMBEDDINGS")
    print("=========================================")

    CHROMA_DB_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(
            CHROMA_DB_PATH
        ),
        collection_metadata={
            "description":
                "Semantic vectors generated from "
                "technical_doc.pdf",

            "source_file":
                PDF_PATH.name
        }
    )

    total_chunks = len(
        document_chunks
    )

    print(
        "Chunks to embed:",
        total_chunks
    )

    print(
        "Embedding batch size:",
        EMBEDDING_BATCH_SIZE
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

        batch_documents = document_chunks[
            batch_start:batch_end
        ]

        batch_ids = chunk_ids[
            batch_start:batch_end
        ]

        print(
            "\nEmbedding chunk batch:",
            f"{batch_start + 1}-{batch_end}",
            f"of {total_chunks}"
        )

        vector_store.add_documents(
            documents=batch_documents,
            ids=batch_ids
        )

        print(
            "Batch stored successfully."
        )

    stored_count = (
        vector_store
        ._collection
        .count()
    )

    print(
        "\nEmbedding generation completed."
    )

    print(
        "ChromaDB directory:"
    )

    print(
        CHROMA_DB_PATH
    )

    print(
        "Collection name:"
    )

    print(
        CHROMA_COLLECTION_NAME
    )

    print(
        "Stored vector records:",
        stored_count
    )

    if stored_count != total_chunks:
        print("\nWARNING:")

        print(
            "The stored-vector count does not match "
            "the document-chunk count."
        )

    return vector_store


# ==========================================================
# 14. REOPEN PERSISTED CHROMADB
# ==========================================================

def reopen_and_validate_collection(
    embedding_model: OpenAIEmbeddings
) -> Chroma:
    """
    Reopen the collection from the persistent local folder.
    """

    print("\n=========================================")
    print("STEP 5: VALIDATING PERSISTED CHROMADB")
    print("=========================================")

    reopened_vector_store = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(
            CHROMA_DB_PATH
        )
    )

    persisted_count = (
        reopened_vector_store
        ._collection
        .count()
    )

    print(
        "Persistent collection reopened successfully."
    )

    print(
        "Persisted vector count:",
        persisted_count
    )

    if persisted_count == 0:
        raise ValueError(
            "The reopened ChromaDB collection "
            "contains no vectors."
        )

    return reopened_vector_store


# ==========================================================
# 15. RUN SEMANTIC SEARCH
# ==========================================================

def run_similarity_search(
    vector_store: Chroma,
    query: str
) -> None:
    """
    Perform semantic search against persisted ChromaDB.
    """

    print("\n=========================================")
    print("STEP 6: SEMANTIC SEARCH VALIDATION")
    print("=========================================")

    print(
        "Search query:"
    )

    print(
        query
    )

    search_results = (
        vector_store
        .similarity_search_with_score(
            query=query,
            k=SEARCH_RESULT_COUNT
        )
    )

    if not search_results:
        print(
            "No semantic-search results were returned."
        )

        return

    print(
        "\nResults returned:",
        len(search_results)
    )

    for result_number, result in enumerate(
        search_results,
        start=1
    ):
        document = result[0]
        score = result[1]

        print("\n-----------------------------------------")

        print(
            f"Result {result_number}"
        )

        print(
            "Similarity distance:",
            score
        )

        print(
            "Source file:",
            document.metadata.get(
                "source_file"
            )
        )

        print(
            "Page number:",
            document.metadata.get(
                "page_number"
            )
        )

        print(
            "Chunk number:",
            document.metadata.get(
                "chunk_number"
            )
        )

        print(
            "Character count:",
            document.metadata.get(
                "character_count"
            )
        )

        print(
            "\nRetrieved content:"
        )

        print(
            document.page_content[:1000]
        )


# ==========================================================
# 16. DISPLAY SUMMARY
# ==========================================================

def print_summary(
    page_count: int,
    chunk_count: int,
    vector_count: int,
    vector_dimension: int
) -> None:
    """
    Display the final Task 8 ingestion summary.
    """

    print("\n=========================================")
    print("TASK 8 COMPLETED SUCCESSFULLY")
    print("=========================================")

    print(
        "Source PDF:",
        PDF_PATH.name
    )

    print(
        "Pages processed:",
        page_count
    )

    print(
        "Chunks created:",
        chunk_count
    )

    print(
        "Vectors stored:",
        vector_count
    )

    print(
        "Vector dimension:",
        vector_dimension
    )

    print(
        "Chat model:",
        OPENAI_CHAT_MODEL
    )

    print(
        "Embedding model:",
        OPENAI_EMBEDDING_MODEL
    )

    print(
        "ChromaDB path:",
        CHROMA_DB_PATH
    )

    print(
        "Collection name:",
        CHROMA_COLLECTION_NAME
    )

    print(
        "\nThe persistent ChromaDB repository is ready "
        "for semantic retrieval and RAG."
    )

    print("=========================================")


# ==========================================================
# 17. MAIN WORKFLOW
# ==========================================================

def main() -> None:
    """
    Execute the complete Task 8 ingestion workflow.
    """

    print("\n=========================================")
    print("ASCEND WEEK 3 - TASK 8")
    print("PDF EMBEDDINGS AND CHROMADB INGESTION")
    print("=========================================")

    validate_configuration()

    page_documents = load_pdf_document(
        PDF_PATH
    )

    document_chunks = split_documents(
        page_documents
    )

    chunk_ids = generate_chunk_ids(
        document_chunks
    )

    embedding_model = create_embedding_model()

    vector_dimension = test_embedding_model(
        embedding_model
    )

    reset_chroma_database()

    create_chroma_vector_store(
        document_chunks=document_chunks,
        chunk_ids=chunk_ids,
        embedding_model=embedding_model
    )

    reopened_vector_store = (
        reopen_and_validate_collection(
            embedding_model
        )
    )

    validation_query = (
        "How should embedding access errors and "
        "ChromaDB retrieval problems be resolved?"
    )

    run_similarity_search(
        vector_store=reopened_vector_store,
        query=validation_query
    )

    stored_vector_count = (
        reopened_vector_store
        ._collection
        .count()
    )

    print_summary(
        page_count=len(page_documents),
        chunk_count=len(document_chunks),
        vector_count=stored_vector_count,
        vector_dimension=vector_dimension
    )


# ==========================================================
# 18. PROGRAM ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    try:
        main()

    except FileNotFoundError as error:
        print("\n=========================================")
        print("FILE ERROR")
        print("=========================================")

        print(error)

    except PermissionError as error:
        print("\n=========================================")
        print("PERMISSION ERROR")
        print("=========================================")

        print(error)

    except Exception as error:
        print("\n=========================================")
        print("TASK 8 FAILED")
        print("=========================================")

        print(
            "Error type:",
            type(error).__name__
        )

        print("\nError details:")

        print(error)

        print("\nTroubleshooting checks:")

        print(
            "1. Confirm technical_doc.pdf exists beside "
            "this Python file."
        )

        print(
            "2. Confirm COPENAI_Embedding_Model contains "
            "an authorized embedding model."
        )

        print(
            "3. Confirm the API key is authorized for "
            "the configured embedding model."
        )

        print(
            "4. Confirm COPEN_AI_BASE_URL supports "
            "the /embeddings endpoint."
        )

        print(
            "5. Confirm check_embedding_ctx_length=False "
            "is present in OpenAIEmbeddings."
        )

        print(
            "6. Confirm langchain-chroma, chromadb, "
            "pypdf, langchain-openai, and "
            "langchain-text-splitters are installed."
        )
        print(
            "7. Confirm the database connection parameters "
            "in the .env file are correct."
        )
        print(
            "8. Confirm the Chroma collection name and database folder "
            "in the .env file are correct."
        )       