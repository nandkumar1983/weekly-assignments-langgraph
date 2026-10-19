import asyncio
import os
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv

from langchain.mcp import MCPAdapter

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables.graph_mermaid import (
    draw_mermaid_png,
)

from langchain_openai import ChatOpenAI

from langgraph.graph import (
    END,
    START,
    StateGraph,
)
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from week4_task7_employee_mcp_server import (
    employee_mcp_server,
)


# ============================================================
# ASCEND PYTHON LANGGRAPH
# WEEK 4 - TASK 7
# EMPLOYEE DIRECTORY ASSISTANT USING LANGGRAPH + MCP
# ============================================================


# ============================================================
# 1. FILE PATH CONFIGURATION
# ============================================================

CURRENT_DIRECTORY = Path(
    __file__
).resolve().parent

ENV_PATH = (
    CURRENT_DIRECTORY
    / ".env"
)

RUNTIME_GRAPH_MERMAID_FILE = (
    CURRENT_DIRECTORY
    / "task7_runtime_graph.mmd"
)

RUNTIME_GRAPH_PNG_FILE = (
    CURRENT_DIRECTORY
    / "task7_runtime_graph.png"
)


# ============================================================
# 2. LOAD ENVIRONMENT
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
    override=True,
)


# ============================================================
# 3. READ LLM CONFIGURATION
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


# ============================================================
# 4. GRAPH SETTINGS
# ============================================================

GRAPH_RECURSION_LIMIT = 12


# ============================================================
# 5. DEFINE LANGGRAPH STATE
# ============================================================

class EmployeeAssistantState(
    TypedDict,
    total=False,
):
    """
    Shared LangGraph state.

    Input:
        question
        messages

    Runtime:
        mcp_tool_used
        mcp_tool_name
        mcp_tool_result
        final_answer
        execution_trace
    """

    question: str

    messages: Annotated[
        list[AnyMessage],
        add_messages,
    ]

    mcp_tool_used: bool

    mcp_tool_name: str

    mcp_tool_result: str

    final_answer: str

    execution_trace: list[str]


# ============================================================
# 6. VALIDATE CONFIGURATION
# ============================================================

def validate_configuration() -> None:
    """
    Validate chat-model configuration.
    """

    required_configuration = {
        "COPEN_AI_BASE_URL":
            OPENAI_BASE_URL,

        "COPEN_AI_API_KEY":
            OPENAI_API_KEY,

        "COPEN_AI_Chat_Model":
            OPENAI_CHAT_MODEL,
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
        "TASK 7 CONFIGURATION"
    )

    print(
        "========================================"
    )

    print(
        "Base URL:",
        OPENAI_BASE_URL,
    )

    print(
        "API Key:",
        "Loaded"
        if OPENAI_API_KEY
        else "Missing",
    )

    print(
        "Chat Model:",
        OPENAI_CHAT_MODEL,
    )

    if missing_configuration:

        raise ValueError(
            "Missing environment variables: "
            + ", ".join(
                missing_configuration
            )
        )


# ============================================================
# 7. CREATE CHAT MODEL
# ============================================================

def create_chat_model() -> ChatOpenAI:
    """
    Create the OpenAI-compatible chat model.
    """

    print(
        "\nInitializing LLM..."
    )

    model = ChatOpenAI(
        model=OPENAI_CHAT_MODEL,

        api_key=OPENAI_API_KEY,

        base_url=OPENAI_BASE_URL,

        temperature=0,

        max_retries=2,

        request_timeout=90,
    )

    print(
        "LLM initialized:"
    )

    print(
        OPENAI_CHAT_MODEL,
    )

    return model


# ============================================================
# 8. SYSTEM PROMPT
# ============================================================

AGENT_SYSTEM_PROMPT = """
You are an Employee Directory Assistant.

Employee information must be obtained only through the
available MCP employee-directory tools.

Rules:

1. When the user asks for an employee's designation,
   manager, or complete employee details, call
   get_employee_details.

2. Pass only the employee name to get_employee_details.

3. When the user asks which employees are available,
   call list_employee_names.

4. Do not invent employee details.

5. Do not answer employee questions using prior knowledge.

6. When an MCP tool result is present, use that result to
   answer the user.

7. Do not call the same tool again after receiving a valid
   tool result.

8. If the employee is not found, state that clearly and
   mention only the available names returned by the tool.

9. Keep the final answer concise and user-friendly.

10. For greetings or general conversation that does not
    request employee information, do not call an MCP tool.
"""


# ============================================================
# 9. RUNTIME COMPONENTS
# ============================================================

base_llm = None

mcp_enabled_llm = None

mcp_tools = None

mcp_tool_node = None


# ============================================================
# 10. LLM AGENT NODE
# ============================================================

def llm_agent_node(
    state: EmployeeAssistantState,
) -> dict:
    """
    Allow the LLM to decide whether an MCP tool is needed.
    """

    if mcp_enabled_llm is None:

        raise RuntimeError(
            "The MCP-enabled LLM has not been initialized."
        )

    messages = list(
        state.get(
            "messages",
            [],
        )
    )

    execution_trace = list(
        state.get(
            "execution_trace",
            [],
        )
    )

    print(
        "\n========================================"
    )

    print(
        "LLM AGENT EXECUTED"
    )

    print(
        "========================================"
    )

    response = mcp_enabled_llm.invoke(
        [
            SystemMessage(
                content=AGENT_SYSTEM_PROMPT,
            ),
            *messages,
        ]
    )

    tool_calls = getattr(
        response,
        "tool_calls",
        [],
    )

    if tool_calls:

        print(
            "LLM decision:"
        )

        print(
            "MCP tool required."
        )

        for tool_call in tool_calls:

            print(
                "MCP tool requested:",
                tool_call.get(
                    "name",
                    "Unknown",
                ),
            )

            print(
                "Arguments:",
                tool_call.get(
                    "args",
                    {},
                ),
            )

        execution_trace.append(
            "LLM Agent requested an MCP employee tool"
        )

    else:

        print(
            "LLM decision:"
        )

        print(
            "Generate the final response."
        )

        execution_trace.append(
            "LLM Agent generated a response"
        )

    return {
        "messages": [response],
        "execution_trace":
            execution_trace,
    }


# ============================================================
# 11. CONDITIONAL ROUTER
# ============================================================

def route_after_llm(
    state: EmployeeAssistantState,
) -> Literal[
    "use_mcp_tool",
    "final_response",
]:
    """
    Route to MCP ToolNode when the latest AI message has
    tool calls.

    Otherwise, route to the final-response node.
    """

    messages = state.get(
        "messages",
        [],
    )

    if not messages:

        return "final_response"

    latest_message = messages[-1]

    tool_calls = getattr(
        latest_message,
        "tool_calls",
        [],
    )

    if tool_calls:

        print(
            "\nRouting:"
        )

        print(
            "LLM Agent -> MCP Employee Tool"
        )

        return "use_mcp_tool"

    print(
        "\nRouting:"
    )

    print(
        "LLM Agent -> Final Answer"
    )

    return "final_response"


# ============================================================
# 12. CAPTURE MCP TOOL RESULT NODE
# ============================================================

def capture_mcp_result_node(
    state: EmployeeAssistantState,
) -> dict:
    """
    Capture the latest MCP ToolMessage after ToolNode runs.
    """

    messages = state.get(
        "messages",
        [],
    )

    execution_trace = list(
        state.get(
            "execution_trace",
            [],
        )
    )

    mcp_tool_name = ""

    mcp_tool_result = ""

    for message in reversed(
        messages
    ):

        if isinstance(
            message,
            ToolMessage,
        ):

            mcp_tool_name = (
                message.name
                or "Unknown MCP Tool"
            )

            mcp_tool_result = str(
                message.content
            )

            break

    print(
        "\n========================================"
    )

    print(
        "MCP TOOL EXECUTED"
    )

    print(
        "========================================"
    )

    print(
        "Tool name:",
        mcp_tool_name,
    )

    print(
        "Tool result:"
    )

    print(
        mcp_tool_result,
    )

    execution_trace.append(
        f"MCP tool executed: {mcp_tool_name}"
    )

    return {
        "mcp_tool_used": True,
        "mcp_tool_name":
            mcp_tool_name,
        "mcp_tool_result":
            mcp_tool_result,
        "execution_trace":
            execution_trace,
    }


# ============================================================
# 13. EXTRACT MESSAGE TEXT
# ============================================================

def extract_message_text(
    message: AIMessage,
) -> str:
    """
    Convert AI message content into plain text.
    """

    content = message.content

    if isinstance(
        content,
        str,
    ):

        return content.strip()

    if isinstance(
        content,
        list,
    ):

        text_parts = []

        for item in content:

            if isinstance(
                item,
                str,
            ):

                text_parts.append(
                    item
                )

            elif isinstance(
                item,
                dict,
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

        return "\n".join(
            text_parts
        ).strip()

    return str(
        content
    ).strip()


# ============================================================
# 14. FINAL RESPONSE NODE
# ============================================================

def final_response_node(
    state: EmployeeAssistantState,
) -> dict:
    """
    Extract the final user-friendly response from the latest
    AI message that does not contain another tool call.
    """

    messages = state.get(
        "messages",
        [],
    )

    execution_trace = list(
        state.get(
            "execution_trace",
            [],
        )
    )

    final_answer = ""

    for message in reversed(
        messages
    ):

        if isinstance(
            message,
            AIMessage,
        ):

            tool_calls = getattr(
                message,
                "tool_calls",
                [],
            )

            if not tool_calls:

                final_answer = (
                    extract_message_text(
                        message
                    )
                )

                break

    if not final_answer:

        final_answer = (
            "I could not produce an answer from the "
            "employee-directory MCP result."
        )

    execution_trace.append(
        "Final employee-directory answer generated"
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
        final_answer,
    )

    return {
        "final_answer":
            final_answer,
        "execution_trace":
            execution_trace,
    }


# ============================================================
# 15. BUILD LANGGRAPH
# ============================================================

def build_employee_graph():
    """
    Build the LangGraph after MCP tools are discovered.

    START
      ↓
    LLM Agent
      ↓
    MCP tool required?
      ├── Yes
      │    ↓
      │ MCP ToolNode
      │    ↓
      │ Capture Result
      │    ↓
      │ LLM Agent
      │
      └── No
           ↓
        Final Answer
           ↓
          END
    """

    global mcp_tool_node

    mcp_tool_node = ToolNode(
        mcp_tools
    )

    graph_builder = StateGraph(
        EmployeeAssistantState
    )

    graph_builder.add_node(
        "llm_agent",
        llm_agent_node,
    )

    graph_builder.add_node(
        "mcp_employee_tool",
        mcp_tool_node,
    )

    graph_builder.add_node(
        "capture_mcp_result",
        capture_mcp_result_node,
    )

    graph_builder.add_node(
        "final_response",
        final_response_node,
    )

    graph_builder.add_edge(
        START,
        "llm_agent",
    )

    graph_builder.add_conditional_edges(
        "llm_agent",
        route_after_llm,
        {
            "use_mcp_tool":
                "mcp_employee_tool",

            "final_response":
                "final_response",
        },
    )

    graph_builder.add_edge(
        "mcp_employee_tool",
        "capture_mcp_result",
    )

    graph_builder.add_edge(
        "capture_mcp_result",
        "llm_agent",
    )

    graph_builder.add_edge(
        "final_response",
        END,
    )

    return graph_builder.compile()


# ============================================================
# 16. CLEAN MERMAID TEXT
# ============================================================

def clean_mermaid_text(
    value: object,
    maximum_length: int = 220,
) -> str:
    """
    Simplify text inserted into Mermaid node labels.
    """

    text_value = str(
        value
    )

    if len(text_value) > maximum_length:

        text_value = (
            text_value[:maximum_length]
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
# 17. BUILD RUNTIME MERMAID
# ============================================================

def build_runtime_mermaid(
    final_state: EmployeeAssistantState,
) -> str:
    """
    Create a runtime graph showing whether the LLM selected
    an MCP tool.
    """

    question = clean_mermaid_text(
        final_state["question"]
    )

    mcp_tool_used = final_state.get(
        "mcp_tool_used",
        False,
    )

    mcp_tool_name = clean_mermaid_text(
        final_state.get(
            "mcp_tool_name",
            "",
        )
    )

    final_answer = clean_mermaid_text(
        final_state["final_answer"]
    )

    if mcp_tool_used:

        runtime_mermaid = f"""
flowchart TD

    START([START])

    QUESTION["User Question: {question}"]

    AGENT["LLM Agent"]

    DECISION{{"Employee MCP Tool Required?"}}

    TOOL["MCP Tool: {mcp_tool_name}"]

    DATA["Employee Data Returned"]

    ANSWER["Final Answer: {final_answer}"]

    END([END])

    START --> QUESTION
    QUESTION --> AGENT
    AGENT --> DECISION
    DECISION -->|Yes| TOOL
    TOOL --> DATA
    DATA --> AGENT
    AGENT --> ANSWER
    ANSWER --> END
"""

    else:

        runtime_mermaid = f"""
flowchart TD

    START([START])

    QUESTION["User Question: {question}"]

    AGENT["LLM Agent"]

    DECISION{{"Employee MCP Tool Required?"}}

    ANSWER["Final Answer: {final_answer}"]

    END([END])

    START --> QUESTION
    QUESTION --> AGENT
    AGENT --> DECISION
    DECISION -->|No| ANSWER
    ANSWER --> END
"""

    return runtime_mermaid.strip()


# ============================================================
# 18. GENERATE RUNTIME GRAPH
# ============================================================

def generate_runtime_graph(
    final_state: EmployeeAssistantState,
) -> None:
    """
    Generate runtime Mermaid and PNG after graph execution.
    """

    print(
        "\n========================================"
    )

    print(
        "GENERATING TASK 7 RUNTIME GRAPH"
    )

    print(
        "========================================"
    )

    runtime_mermaid = (
        build_runtime_mermaid(
            final_state
        )
    )

    RUNTIME_GRAPH_MERMAID_FILE.write_text(
        runtime_mermaid,
        encoding="utf-8",
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

                print(
                    "Runtime graph opened automatically."
                )

            except OSError as open_error:

                print(
                    "PNG generated but Windows could not "
                    "open the file automatically."
                )

                print(
                    "Open error:",
                    open_error,
                )

    except Exception as graph_error:

        print(
            "\nPNG rendering failed."
        )

        print(
            "Error type:",
            type(graph_error).__name__,
        )

        print(
            "Error details:",
            graph_error,
        )

        print(
            "\nThe Mermaid graph remains available:"
        )

        print(
            RUNTIME_GRAPH_MERMAID_FILE
        )


# ============================================================
# 19. EXECUTE ONE EMPLOYEE QUESTION
# ============================================================

async def answer_employee_question(
    employee_graph,
    question: str,
) -> EmployeeAssistantState:
    """
    Execute one employee-directory question.
    """

    input_state: EmployeeAssistantState = {
        "question":
            question,

        "messages":
            [
                HumanMessage(
                    content=question,
                )
            ],

        "mcp_tool_used":
            False,

        "mcp_tool_name":
            "",

        "mcp_tool_result":
            "",

        "final_answer":
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
        "User question:"
    )

    print(
        question,
    )

    final_state = await employee_graph.ainvoke(
        input_state,
        config={
            "recursion_limit":
                GRAPH_RECURSION_LIMIT,
        },
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
        final_state.get(
            "execution_trace",
            [],
        ),
        start=1,
    ):

        print(
            f"{trace_number}. {trace_message}"
        )

    print(
        "\n========================================"
    )

    print(
        "FINAL RESPONSE"
    )

    print(
        "========================================"
    )

    print(
        final_state["final_answer"]
    )

    generate_runtime_graph(
        final_state
    )

    return final_state


# ============================================================
# 20. DISPLAY HELP
# ============================================================

def display_help() -> None:
    """
    Display useful Task 7 questions.
    """

    print(
        "\n========================================"
    )

    print(
        "EXAMPLE EMPLOYEE QUESTIONS"
    )

    print(
        "========================================"
    )

    print(
        "1. Who is the manager of Vaishali?"
    )

    print(
        "2. Find the designation of Rupali Giri."
    )

    print(
        "3. Give me all details for Vaishali."
    )

    print(
        "4. Who is Rupali Giri's manager?"
    )

    print(
        "5. Which employees are available?"
    )

    print(
        "6. Find the designation of an unknown employee."
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
# 21. ASYNC MAIN
# ============================================================

async def main() -> None:
    """
    Connect to the MCP server, discover MCP tools, build the
    LangGraph, and start the interactive assistant.
    """

    global base_llm
    global mcp_enabled_llm
    global mcp_tools

    print(
        "\n========================================"
    )

    print(
        "ASCEND PYTHON LANGGRAPH"
    )

    print(
        "WEEK 4 - TASK 7"
    )

    print(
        "EMPLOYEE DIRECTORY ASSISTANT"
    )

    print(
        "LANGGRAPH + LLM + MCP"
    )

    print(
        "========================================"
    )

    validate_configuration()

    base_llm = create_chat_model()

    print(
        "\nConnecting to Employee Directory "
        "MCP Server..."
    )

    # The FastMCP server runs in-process.
    # The MCP adapter performs dynamic tool discovery.

    async with MCPAdapter(
        employee_mcp_server
    ) as adapter:

        mcp_tools = await adapter.list_tools()

        if not mcp_tools:

            raise RuntimeError(
                "No MCP tools were discovered."
            )

        print(
            "MCP connection successful."
        )

        print(
            "MCP tools discovered:",
            len(mcp_tools),
        )

        for tool_number, discovered_tool in enumerate(
            mcp_tools,
            start=1,
        ):

            print(
                f"\nTool {tool_number}:"
            )

            print(
                "Name:",
                discovered_tool.name,
            )

            print(
                "Description:"
            )

            print(
                discovered_tool.description,
            )

        # Bind dynamically discovered MCP tools to the LLM.

        mcp_enabled_llm = base_llm.bind_tools(
            mcp_tools
        )

        print(
            "\nMCP tools bound to the LLM."
        )

        employee_graph = (
            build_employee_graph()
        )

        print(
            "LangGraph created successfully."
        )

        display_help()

        while True:

            print(
                "\n----------------------------------------"
            )

            question = input(
                "Ask an employee question: "
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
                    "\nTask 7 Employee Directory "
                    "Assistant stopped."
                )

                break

            if normalized_question in [
                "help",
                "show help",
            ]:

                display_help()

                continue

            try:

                await answer_employee_question(
                    employee_graph=
                        employee_graph,

                    question=
                        question,
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
                    type(question_error).__name__,
                )

                print(
                    "Error details:",
                    question_error,
                )


# ============================================================
# 22. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        asyncio.run(
            main()
        )

    except KeyboardInterrupt:

        print(
            "\n\nTask 7 stopped by user."
        )

    except Exception as application_error:

        print(
            "\n========================================"
        )

        print(
            "TASK 7 FAILED"
        )

        print(
            "========================================"
        )

        print(
            "Error type:",
            type(application_error).__name__,
        )

        print(
            "Error details:",
            application_error,
        )
        print(
            "========================================"
        )