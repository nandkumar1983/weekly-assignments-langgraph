import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastmcp import FastMCP


# ==========================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ==========================================================

load_dotenv(
    override=True
)


# ==========================================================
# 2. READ MCP SERVER CONFIGURATION
# ==========================================================

SERVER_HOST = os.getenv(
    "MCP_SERVER_HOST",
    "127.0.0.1"
)

SERVER_PORT = int(
    os.getenv(
        "MCP_SERVER_PORT",
        "8000"
    )
)


# ==========================================================
# 3. CREATE FASTMCP SERVER
# ==========================================================

mcp = FastMCP(
    "Ascend Week 3 MCP Utility Server"
)


# ==========================================================
# 4. MCP TOOL - CURRENT SERVER TIMESTAMP
# ==========================================================

@mcp.tool
def get_current_server_timestamp() -> str:
    """
    Return the current UTC timestamp from the MCP server.

    Use this tool whenever the user asks for the current
    server timestamp, server time, current date and time,
    or current UTC time.
    """

    timestamp = datetime.now(
        timezone.utc
    ).isoformat(
        timespec="seconds"
    )

    print(
        "\n=========================================="
    )

    print(
        "[MCP SERVER] TOOL EXECUTED"
    )

    print(
        "Tool   : get_current_server_timestamp"
    )

    print(
        "Result :",
        timestamp
    )

    print(
        "=========================================="
    )

    return timestamp


# ==========================================================
# 5. MCP TOOL - ADD TWO INTEGERS
# ==========================================================

@mcp.tool
def add_two_integers(
    first_number: int,
    second_number: int
) -> int:
    """
    Add two integers and return their sum.

    Use this tool whenever the user asks to add,
    sum, total, or calculate two integer values.

    Args:
        first_number:
            First integer.

        second_number:
            Second integer.

    Returns:
        Sum of the two integers.
    """

    result = (
        first_number
        +
        second_number
    )

    print(
        "\n=========================================="
    )

    print(
        "[MCP SERVER] TOOL EXECUTED"
    )

    print(
        "Tool   : add_two_integers"
    )

    print(
        "Input  :",
        first_number,
        "+",
        second_number
    )

    print(
        "Result :",
        result
    )

    print(
        "=========================================="
    )

    return result


# ==========================================================
# 6. START MCP SERVER
# ==========================================================

def main():

    print(
        "\n================================================"
    )

    print(
        " ASCEND WEEK 3 - TASK 7"
    )

    print(
        " MCP SERVER USING FASTMCP"
    )

    print(
        "================================================"
    )

    print(
        "\nServer Name:"
    )

    print(
        "Ascend Week 3 MCP Utility Server"
    )

    print(
        "\nHost:",
        SERVER_HOST
    )

    print(
        "Port:",
        SERVER_PORT
    )

    print(
        "Transport: SSE"
    )

    print(
        "\nSSE Endpoint:"
    )

    print(
        f"http://{SERVER_HOST}:{SERVER_PORT}/sse"
    )

    print(
        "\nPublished MCP Tools:"
    )

    print(
        "1. get_current_server_timestamp"
    )

    print(
        "2. add_two_integers"
    )

    print(
        "\nStarting MCP Server..."
    )

    print(
        "Keep this terminal running."
    )

    # Assignment explicitly requires SSE.
    mcp.run(
        transport="sse",
        host=SERVER_HOST,
        port=SERVER_PORT
    )


# ==========================================================
# 7. PROGRAM ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()