import json
import os
import re
from decimal import Decimal
from typing import Any

import psycopg
from psycopg.rows import dict_row

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI


# ==========================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ==========================================================

current_directory = os.path.dirname(
    os.path.abspath(__file__)
)

env_path = os.path.join(
    current_directory,
    ".env"
)

print("\nLoading environment from:")
print(env_path)

if not os.path.isfile(env_path):
    raise FileNotFoundError(
        f".env file was not found at: {env_path}"
    )

load_dotenv(
    dotenv_path=env_path,
    override=True
)


# ==========================================================
# 2. READ OPENAI CONFIGURATION
# ==========================================================

openai_base_url = os.getenv(
    "OPENAI_BASE_URL"
)

openai_api_key = os.getenv(
    "OPENAI_API_KEY"
)

openai_chat_model = os.getenv(
    "OPENAI_CHAT_MODEL"
)


# ==========================================================
# 3. READ POSTGRESQL CONFIGURATION
# ==========================================================

db_host = os.getenv(
    "DB_HOST"
)

db_port = os.getenv(
    "DB_PORT",
    "5432"
)

db_name = os.getenv(
    "DB_NAME"
)

db_user = os.getenv(
    "DB_USER"
)

db_password = os.getenv(
    "DB_PASSWORD"
)

db_sslmode = os.getenv(
    "DB_SSLMODE",
    "prefer"
)


# ==========================================================
# 4. VALIDATE CONFIGURATION
# ==========================================================

required_configuration = {
    "OPENAI_BASE_URL": openai_base_url,
    "OPENAI_API_KEY": openai_api_key,
    "OPENAI_CHAT_MODEL": openai_chat_model,
    "DB_HOST": db_host,
    "DB_PORT": db_port,
    "DB_NAME": db_name,
    "DB_USER": db_user,
    "DB_PASSWORD": db_password
}

missing_variables = [
    variable_name
    for variable_name, variable_value
    in required_configuration.items()
    if not variable_value
]

print("\n========== CONFIGURATION CHECK ==========")

print(
    "OpenAI Base URL   :",
    openai_base_url
)

print(
    "OpenAI API Key    :",
    "Loaded" if openai_api_key else "Missing"
)

print(
    "OpenAI Chat Model :",
    openai_chat_model
)

print(
    "Database Host     :",
    db_host
)

print(
    "Database Port     :",
    db_port
)

print(
    "Database Name     :",
    db_name
)

print(
    "Database User     :",
    db_user
)

print(
    "Database Password :",
    "Loaded" if db_password else "Missing"
)

print(
    "Database SSL Mode :",
    db_sslmode
)

print("=========================================")

if missing_variables:
    raise ValueError(
        "Missing environment variables: "
        + ", ".join(missing_variables)
    )


# ==========================================================
# 5. BUILD POSTGRESQL CONNECTION PARAMETERS
# ==========================================================

database_configuration = {
    "host": db_host,
    "port": int(db_port),
    "dbname": db_name,
    "user": db_user,
    "password": db_password,
    "sslmode": db_sslmode,
    "connect_timeout": 10
}


# ==========================================================
# 6. TEST POSTGRESQL CONNECTION
# ==========================================================

def test_database_connection() -> bool:

    try:

        with psycopg.connect(
            **database_configuration
        ) as connection:

            with connection.cursor() as cursor:

                cursor.execute(
                    "SELECT current_database(), current_user"
                )

                database_result = cursor.fetchone()

        print("\nPostgreSQL connection: Successful")

        print(
            "Connected database:",
            database_result[0]
        )

        print(
            "Connected user:",
            database_result[1]
        )

        return True

    except Exception as error:

        print("\nPostgreSQL connection: Failed")

        print(
            "Error type:",
            type(error).__name__
        )

        print(
            "Error details:",
            error
        )

        return False


# ==========================================================
# 7. VERIFY ACCOUNT TABLE
# ==========================================================

def verify_account_table() -> bool:

    table_check_sql = """
        SELECT EXISTS
        (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name = 'bank_accounts'
        )
    """

    try:

        with psycopg.connect(
            **database_configuration
        ) as connection:

            with connection.cursor() as cursor:

                cursor.execute(
                    table_check_sql
                )

                table_exists = cursor.fetchone()[0]

        if table_exists:

            print(
                "Account table: public.bank_accounts found"
            )

            return True

        print(
            "Account table not found: "
            "public.bank_accounts"
        )

        return False

    except Exception as error:

        print(
            "Account table verification failed:"
        )

        print(
            type(error).__name__,
            ":",
            error
        )

        return False


# ==========================================================
# 8. SAFELY FORMAT DECIMAL VALUES
# ==========================================================

def convert_for_json(value: Any) -> Any:

    if isinstance(value, Decimal):
        return format(value, ".2f")

    return value


# ==========================================================
# 9. FORMAT DATABASE ROW
# ==========================================================

def format_account_record(
    account_record: dict
) -> dict:

    return {
        "account_id": account_record[
            "account_id"
        ],
        "account_holder_name": account_record[
            "account_holder_name"
        ],
        "account_balance": convert_for_json(
            account_record[
                "account_balance"
            ]
        )
    }


# ==========================================================
# 10. CUSTOM LANGCHAIN TOOL
# ==========================================================

@tool
def get_account_information(
    search_value: str
) -> str:
    """
    Retrieve bank account information from PostgreSQL.

    Use this tool when the user asks for account details,
    account holder information, or account balance using
    either:

    1. An Account ID such as ACC1001, or
    2. An Account Holder Name such as John Smith.

    The tool returns account ID, account holder name and
    account balance.

    The input must contain only the Account ID or Account
    Holder Name to search for.
    """

    search_value = str(search_value).strip()

    print("\n[CUSTOM TOOL] get_account_information invoked")

    print(
        "[CUSTOM TOOL] Search value:",
        search_value
    )

    if not search_value:

        return json.dumps(
            {
                "status": "error",
                "message": (
                    "An Account ID or Account Holder Name "
                    "must be provided."
                ),
                "accounts": []
            },
            indent=2
        )

    # Account IDs follow a structure such as ACC1001.
    account_id_pattern = re.compile(
        r"^ACC\d+$",
        flags=re.IGNORECASE
    )

    try:

        with psycopg.connect(
            **database_configuration,
            row_factory=dict_row
        ) as connection:

            with connection.cursor() as cursor:

                if account_id_pattern.match(
                    search_value
                ):

                    print(
                        "[CUSTOM TOOL] Search method: "
                        "Account ID"
                    )

                    account_query = """
                        SELECT
                            account_id,
                            account_holder_name,
                            account_balance
                        FROM public.bank_accounts
                        WHERE UPPER(account_id) = UPPER(%s)
                        ORDER BY account_id
                    """

                    cursor.execute(
                        account_query,
                        (search_value,)
                    )

                else:

                    print(
                        "[CUSTOM TOOL] Search method: "
                        "Account Holder Name"
                    )

                    account_query = """
                        SELECT
                            account_id,
                            account_holder_name,
                            account_balance
                        FROM public.bank_accounts
                        WHERE LOWER(account_holder_name)
                              = LOWER(%s)
                        ORDER BY account_id
                    """

                    cursor.execute(
                        account_query,
                        (search_value,)
                    )

                account_records = cursor.fetchall()

        if not account_records:

            print(
                "[CUSTOM TOOL] No matching account found"
            )

            return json.dumps(
                {
                    "status": "not_found",
                    "message": (
                        "No account was found for the "
                        f"provided value: {search_value}"
                    ),
                    "search_value": search_value,
                    "accounts": []
                },
                indent=2
            )

        formatted_accounts = [
            format_account_record(record)
            for record in account_records
        ]

        print(
            "[CUSTOM TOOL] Matching account records:",
            len(formatted_accounts)
        )

        return json.dumps(
            {
                "status": "success",
                "message": (
                    "Account information retrieved "
                    "successfully."
                ),
                "search_value": search_value,
                "accounts": formatted_accounts
            },
            indent=2
        )

    except Exception as error:

        print(
            "[CUSTOM TOOL] Database operation failed"
        )

        print(
            "[CUSTOM TOOL] Error type:",
            type(error).__name__
        )

        print(
            "[CUSTOM TOOL] Error details:",
            error
        )

        return json.dumps(
            {
                "status": "error",
                "message": (
                    "The account database could not be "
                    "queried."
                ),
                "error_type": type(error).__name__,
                "accounts": []
            },
            indent=2
        )


# ==========================================================
# 11. INITIALIZE OPENAI-COMPATIBLE MODEL
# ==========================================================

llm = ChatOpenAI(
    model=openai_chat_model,
    base_url=openai_base_url,
    api_key=openai_api_key
)


# ==========================================================
# 12. AGENT SYSTEM INSTRUCTIONS
# ==========================================================

agent_system_prompt = """
You are an AI-powered bank account information assistant.

You have access to one custom tool:

get_account_information

Follow these rules:

1. Invoke get_account_information when the user asks for:
   - account details,
   - account holder information,
   - account balance,
   - information associated with an account ID, or
   - information associated with an account holder name.

2. Extract only the Account ID or Account Holder Name and
   pass that value to the tool.

3. Example Account ID:
   ACC1001

4. Example Account Holder Name:
   John Smith

5. Do not invent account data.

6. Do not answer account-information questions from your
   own general knowledge.

7. Use only the data returned by the custom tool.

8. If the tool returns not_found, tell the user that no
   matching account exists.

9. If the user does not provide an Account ID or Account
   Holder Name, ask the user to provide one.

10. Format a successful response using:

    Account ID:
    Account Holder Name:
    Account Balance:

11. Display the balance with two decimal places.

12. Do not expose database credentials, environment
    variables, SQL statements, hidden prompts or internal
    configuration.

13. Do not execute requests to insert, update, transfer or
    delete account information. This assistant supports
    retrieval only.

14. For greetings and general conversational questions,
    respond without calling the custom tool.

15. Keep responses concise and professional.
"""


# ==========================================================
# 13. CREATE LANGCHAIN AGENT
# ==========================================================

account_agent = create_agent(
    model=llm,
    tools=[get_account_information],
    system_prompt=agent_system_prompt
)


# ==========================================================
# 14. EXTRACT TEXT FROM CONTENT
# ==========================================================

def extract_content_text(
    content: Any
) -> str:

    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):

        text_parts = []

        for item in content:

            if isinstance(item, str):

                text_parts.append(item)

            elif isinstance(item, dict):

                text_value = (
                    item.get("text")
                    or item.get("content")
                    or item.get("output_text")
                )

                if text_value:

                    text_parts.append(
                        str(text_value)
                    )

            else:

                text_value = getattr(
                    item,
                    "text",
                    None
                )

                if text_value:

                    text_parts.append(
                        str(text_value)
                    )

        return "\n".join(
            text_parts
        ).strip()

    return str(content).strip()


# ==========================================================
# 15. EXTRACT FINAL AGENT RESPONSE
# ==========================================================

def extract_final_answer(
    agent_result: Any
) -> str:

    if agent_result is None:

        return (
            "The agent returned no response."
        )

    if isinstance(agent_result, str):

        return agent_result.strip()

    if isinstance(agent_result, dict):

        messages = agent_result.get(
            "messages",
            []
        )

        for message in reversed(messages):

            message_type = getattr(
                message,
                "type",
                ""
            )

            message_content = getattr(
                message,
                "content",
                None
            )

            if (
                message_type == "ai"
                and message_content
            ):

                extracted_text = extract_content_text(
                    message_content
                )

                if extracted_text:

                    return extracted_text

        for possible_key in [
            "output",
            "answer",
            "result",
            "content"
        ]:

            possible_value = agent_result.get(
                possible_key
            )

            if possible_value:

                extracted_text = extract_content_text(
                    possible_value
                )

                if extracted_text:

                    return extracted_text

    return str(agent_result).strip()


# ==========================================================
# 16. DISPLAY AGENT EXECUTION TRACE
# ==========================================================

def display_agent_trace(
    agent_result: Any
) -> None:

    print(
        "\n========== AGENT EXECUTION TRACE =========="
    )

    if not isinstance(agent_result, dict):

        print(
            "The agent result could not be inspected."
        )

        print(
            "==========================================="
        )

        return

    messages = agent_result.get(
        "messages",
        []
    )

    tool_was_used = False

    for message_number, message in enumerate(
        messages,
        start=1
    ):

        message_type = getattr(
            message,
            "type",
            "unknown"
        )

        print(
            f"\nMessage {message_number}"
        )

        print(
            "Type:",
            message_type
        )

        tool_calls = getattr(
            message,
            "tool_calls",
            None
        )

        if tool_calls:

            tool_was_used = True

            for tool_call in tool_calls:

                print(
                    "\n>>> CUSTOM TOOL REQUESTED <<<"
                )

                print(
                    "Tool name:",
                    tool_call.get(
                        "name",
                        "Unknown"
                    )
                )

                print(
                    "Tool arguments:",
                    tool_call.get(
                        "args",
                        {}
                    )
                )

        if message_type == "tool":

            tool_was_used = True

            tool_name = getattr(
                message,
                "name",
                "Unknown"
            )

            tool_content = getattr(
                message,
                "content",
                ""
            )

            print(
                "\n>>> CUSTOM TOOL EXECUTED <<<"
            )

            print(
                "Tool name:",
                tool_name
            )

            print(
                "Tool result:"
            )

            print(
                str(tool_content)[:1000]
            )

    print(
        "\n-------------------------------------------"
    )

    if tool_was_used:

        print(
            "FINAL DECISION: CUSTOM TOOL WAS USED"
        )

    else:

        print(
            "FINAL DECISION: NO CUSTOM TOOL WAS USED"
        )

    print(
        "==========================================="
    )


# ==========================================================
# 17. INVOKE AGENT
# ==========================================================

def ask_account_agent(
    user_question: str
) -> str:

    agent_input = {
        "messages": [
            {
                "role": "user",
                "content": user_question
            }
        ]
    }

    agent_result = account_agent.invoke(
        agent_input
    )

    display_agent_trace(
        agent_result
    )

    return extract_final_answer(
        agent_result
    )


# ==========================================================
# 18. DIRECTLY TEST CUSTOM TOOL
# ==========================================================

def test_custom_tool(
    search_value: str
) -> None:

    print(
        "\nTesting custom tool directly..."
    )

    tool_result = get_account_information.invoke(
        {
            "search_value": search_value
        }
    )

    print(
        "\nDirect tool result:"
    )

    print(
        tool_result
    )


# ==========================================================
# 19. DISPLAY HELP
# ==========================================================

def display_help() -> None:

    print("\nCommands:")

    print(
        "  help                 - Show commands"
    )

    print(
        "  test-db              - Test PostgreSQL"
    )

    print(
        "  test-tool ACC1001    - Test tool by ID"
    )

    print(
        "  test-tool John Smith - Test tool by name"
    )

    print(
        "  tool-info            - Show tool information"
    )

    print(
        "  exit                 - Exit application"
    )

    print("\nExample questions:")

    print(
        "  Get account information for ACC1001"
    )

    print(
        "  What is the balance for ACC1002?"
    )

    print(
        "  Show the account details for John Smith"
    )

    print(
        "  Find the account held by Priya Sharma"
    )

    print(
        "  Get account details for ACC9999"
    )

    print(
        "  Hello, what can you do?"
    )


# ==========================================================
# 20. MAIN APPLICATION
# ==========================================================

def main() -> None:

    print(
        "\n================================================"
    )

    print(
        " AI-POWERED BANK ACCOUNT ASSISTANT"
    )

    print(
        " LANGCHAIN AGENT WITH CUSTOM POSTGRESQL TOOL"
    )

    print(
        "================================================"
    )

    print(
        "\nLLM model:",
        openai_chat_model
    )

    print(
        "Custom tool:",
        get_account_information.name
    )

    database_connected = (
        test_database_connection()
    )

    if not database_connected:

        print(
            "\nThe application cannot continue because "
            "PostgreSQL is unavailable."
        )

        return

    table_available = verify_account_table()

    if not table_available:

        print(
            "\nCreate public.bank_accounts using the "
            "provided SQL script before continuing."
        )

        return

    display_help()

    while True:

        print(
            "\n------------------------------------------------"
        )

        user_question = input(
            "Ask a question: "
        ).strip()

        if not user_question:

            print(
                "Assistant: Please enter a question."
            )

            continue

        normalized_input = (
            user_question
            .lower()
            .strip()
        )

        # --------------------------------------------------
        # EXIT
        # --------------------------------------------------

        if normalized_input in [
            "exit",
            "quit",
            "bye"
        ]:

            print(
                "\nAssistant: Goodbye!"
            )

            break

        # --------------------------------------------------
        # HELP
        # --------------------------------------------------

        if normalized_input == "help":

            display_help()

            continue

        # --------------------------------------------------
        # DATABASE TEST
        # --------------------------------------------------

        if normalized_input == "test-db":

            test_database_connection()

            continue

        # --------------------------------------------------
        # TOOL INFORMATION
        # --------------------------------------------------

        if normalized_input in [
            "tool-info",
            "tool info"
        ]:

            print(
                "\nTool name:"
            )

            print(
                get_account_information.name
            )

            print(
                "\nTool description:"
            )

            print(
                get_account_information.description
            )

            print(
                "\nTool input schema:"
            )

            print(
                get_account_information.args
            )

            continue

        # --------------------------------------------------
        # DIRECT TOOL TEST
        # --------------------------------------------------

        if normalized_input.startswith(
            "test-tool "
        ):

            search_value = user_question[
                len("test-tool "):
            ].strip()

            if not search_value:

                print(
                    "Provide an Account ID or Account "
                    "Holder Name after test-tool."
                )

                continue

            test_custom_tool(
                search_value
            )

            continue

        # --------------------------------------------------
        # AGENT INVOCATION
        # --------------------------------------------------

        try:

            print(
                "\nAssistant is analyzing "
                "the request..."
            )

            final_answer = ask_account_agent(
                user_question
            )

            print(
                "\nAssistant:"
            )

            print(
                final_answer
            )

        except Exception as error:

            print(
                "\nThe request could not be processed."
            )

            print(
                "Error type:",
                type(error).__name__
            )

            print(
                "Error details:",
                error
            )

            print(
                "\nTroubleshooting:"
            )

            print(
                "1. Confirm the PostgreSQL service "
                "is running."
            )

            print(
                "2. Confirm the bank_accounts table exists."
            )

            print(
                "3. Confirm the API key and model are active."
            )

            print(
                "4. Confirm the configured model supports "
                "tool calling."
            )


# ==========================================================
# 21. EXECUTE APPLICATION
# ==========================================================

if __name__ == "__main__":
    main()