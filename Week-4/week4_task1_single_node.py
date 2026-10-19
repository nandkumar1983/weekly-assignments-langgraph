import os
import re
from pathlib import Path
from typing import TypedDict

from langchain_core.runnables.graph_mermaid import (
    draw_mermaid_png
)
from langgraph.graph import StateGraph, START, END


# ============================================================
# ASCEND PYTHON LANGGRAPH
# WEEK 4 - TASK 1
# SINGLE NODE GRAPH
# ============================================================


# ============================================================
# 1. FILE PATH CONFIGURATION
# ============================================================

CURRENT_DIRECTORY = Path(
    __file__
).resolve().parent

RUNTIME_GRAPH_MERMAID_FILE = (
    CURRENT_DIRECTORY
    / "task1_runtime_graph.mmd"
)

RUNTIME_GRAPH_PNG_FILE = (
    CURRENT_DIRECTORY
    / "task1_runtime_graph.png"
)

COMPILED_GRAPH_MERMAID_FILE = (
    CURRENT_DIRECTORY
    / "task1_compiled_graph.mmd"
)


# ============================================================
# 2. DEFINE LANGGRAPH STATE
# ============================================================

class CalculatorState(TypedDict, total=False):
    """
    State passed through the single-node LangGraph.

    Input fields:
        name
        values
        operation

    Output fields:
        result
        message
    """

    name: str
    values: list[int]
    operation: str
    result: int
    message: str


# ============================================================
# 3. SINGLE PROCESSING NODE
# ============================================================

def calculate_sum_or_product(
    state: CalculatorState
) -> dict:
    """
    The only processing node in the graph.

    If operation is '+', add all values.

    If operation is '*', multiply all values.
    """

    name = state["name"]
    values = state["values"]
    operation = state["operation"]

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not name.strip():
        raise ValueError(
            "Name cannot be empty."
        )

    if not values:
        raise ValueError(
            "At least one number is required."
        )

    # --------------------------------------------------------
    # Addition
    # --------------------------------------------------------

    if operation == "+":
        result = sum(values)

    # --------------------------------------------------------
    # Multiplication
    # --------------------------------------------------------

    elif operation == "*":
        result = 1

        for number in values:
            result *= number

    # --------------------------------------------------------
    # Invalid operation
    # --------------------------------------------------------

    else:
        raise ValueError(
            "Operation must be '+' or '*'."
        )

    message = (
        f"Hi {name}, your answer is: {result}"
    )

    return {
        "result": result,
        "message": message
    }


# ============================================================
# 4. BUILD THE SINGLE-NODE LANGGRAPH
# ============================================================

graph_builder = StateGraph(
    CalculatorState
)

NODE_NAME = "calculate_sum_or_product"

graph_builder.add_node(
    NODE_NAME,
    calculate_sum_or_product
)

graph_builder.add_edge(
    START,
    NODE_NAME
)

graph_builder.add_edge(
    NODE_NAME,
    END
)


# ============================================================
# 5. COMPILE LANGGRAPH
# ============================================================

calculator_graph = (
    graph_builder.compile()
)


# ============================================================
# 6. SAVE COMPILED GRAPH DEFINITION
#
# This does not generate the runtime PNG.
# It only stores the actual compiled LangGraph structure.
# ============================================================

def save_compiled_graph_definition() -> None:
    """
    Save the real compiled LangGraph structure as Mermaid.

    The runtime PNG is generated separately after each
    operation.
    """

    compiled_mermaid = (
        calculator_graph
        .get_graph()
        .draw_mermaid()
    )

    COMPILED_GRAPH_MERMAID_FILE.write_text(
        compiled_mermaid,
        encoding="utf-8"
    )


# ============================================================
# 7. PARSE NATURAL-LANGUAGE OPERATION
# ============================================================

def parse_user_command(
    user_command: str
) -> tuple[list[int], str]:
    """
    Convert user text into values and operation.

    Examples:

        add 9 and 5

        sum 10 20 30

        multiply 4 and 6

        product of 9 and 6

        product 5 5 5
    """

    normalized_command = (
        user_command
        .strip()
        .lower()
    )

    # --------------------------------------------------------
    # Detect addition
    # --------------------------------------------------------

    if (
        normalized_command.startswith("add")
        or normalized_command.startswith("sum")
        or normalized_command.startswith("+")
    ):
        operation = "+"

    # --------------------------------------------------------
    # Detect multiplication
    # --------------------------------------------------------

    elif (
        normalized_command.startswith("multiply")
        or normalized_command.startswith("product")
        or normalized_command.startswith("times")
        or normalized_command.startswith("*")
    ):
        operation = "*"

    # --------------------------------------------------------
    # Unknown operation
    # --------------------------------------------------------

    else:
        raise ValueError(
            "Start the command with 'add', 'sum', "
            "'multiply', or 'product'."
        )

    # --------------------------------------------------------
    # Extract positive or negative integer values
    # --------------------------------------------------------

    number_strings = re.findall(
        r"-?\d+",
        normalized_command
    )

    if not number_strings:
        raise ValueError(
            "No numbers were found in the command."
        )

    values = [
        int(number)
        for number in number_strings
    ]

    if len(values) < 2:
        raise ValueError(
            "Provide at least two numbers."
        )

    return values, operation


# ============================================================
# 8. FORMAT OPERATION FOR DISPLAY
# ============================================================

def format_operation_expression(
    values: list[int],
    operation: str
) -> str:
    """
    Create a readable arithmetic expression.
    """

    separator = (
        " + "
        if operation == "+"
        else " × "
    )

    return separator.join(
        str(value)
        for value in values
    )


# ============================================================
# 9. ESCAPE TEXT FOR MERMAID LABELS
# ============================================================

def escape_mermaid_text(
    value: object
) -> str:
    """
    Escape characters that may interfere with Mermaid.
    """

    text_value = str(value)

    return (
        text_value
        .replace("&", "&amp;")
        .replace('"', "'")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", " ")
    )


# ============================================================
# 10. BUILD RUNTIME-SPECIFIC MERMAID GRAPH
# ============================================================

def build_runtime_mermaid(
    input_state: CalculatorState,
    output_state: CalculatorState
) -> str:
    """
    Build a runtime graph containing the actual operation.

    This diagram still has exactly one processing node:

        START
          |
          v
        Calculate Sum or Product
          |
          v
         END
    """

    name = escape_mermaid_text(
        input_state["name"]
    )

    values = input_state["values"]
    operation = input_state["operation"]
    result = output_state["result"]

    expression = format_operation_expression(
        values,
        operation
    )

    expression = escape_mermaid_text(
        expression
    )

    operation_name = (
        "Addition"
        if operation == "+"
        else "Multiplication"
    )

    output_message = escape_mermaid_text(
        output_state["message"]
    )

    runtime_mermaid = f"""
---
config:
  flowchart:
    curve: linear
---
flowchart TD

    START(["START"])

    CALCULATOR["<b>Calculate Sum or Product</b><br/><br/>Name: {name}<br/>Values: {values}<br/>Operation: {operation_name}<br/><br/>{expression} = {result}<br/><br/>{output_message}"]

    END(["END"])

    START --> CALCULATOR
    CALCULATOR --> END

    classDef startNode fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#1B5E20;
    classDef processNode fill:#E3F2FD,stroke:#1565C0,stroke-width:2px,color:#0D47A1;
    classDef endNode fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px,color:#4A148C;

    class START startNode;
    class CALCULATOR processNode;
    class END endNode;
""".strip()

    return runtime_mermaid


# ============================================================
# 11. GENERATE RUNTIME GRAPH AFTER OPERATION
# ============================================================

def generate_runtime_graph(
    input_state: CalculatorState,
    output_state: CalculatorState
) -> None:
    """
    Generate and open the graph only after the current
    arithmetic operation has executed.
    """

    print(
        "\n========================================"
    )

    print(
        "GENERATING RUNTIME GRAPH"
    )

    print(
        "========================================"
    )

    runtime_mermaid = build_runtime_mermaid(
        input_state=input_state,
        output_state=output_state
    )

    # --------------------------------------------------------
    # Save runtime Mermaid source
    # --------------------------------------------------------

    RUNTIME_GRAPH_MERMAID_FILE.write_text(
        runtime_mermaid,
        encoding="utf-8"
    )

    print(
        "\nRuntime Mermaid graph:"
    )

    print(
        runtime_mermaid
    )

    print(
        "\nMermaid file saved:"
    )

    print(
        RUNTIME_GRAPH_MERMAID_FILE
    )

    # --------------------------------------------------------
    # Render runtime Mermaid as PNG
    # --------------------------------------------------------

    try:
        png_data = draw_mermaid_png(
            mermaid_syntax=runtime_mermaid,
            background_color="white",
            padding=20,
            max_retries=2,
            retry_delay=1.0
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

        # ----------------------------------------------------
        # Open PNG automatically on Windows
        # ----------------------------------------------------

        if os.name == "nt":
            try:
                os.startfile(
                    str(
                        RUNTIME_GRAPH_PNG_FILE
                    )
                )

                print(
                    "\nRuntime graph opened automatically."
                )

            except OSError as open_error:
                print(
                    "\nPNG was generated, but Windows "
                    "could not open it automatically."
                )

                print(
                    "Open error:",
                    open_error
                )

    except Exception as graph_error:
        print(
            "\nRuntime PNG generation failed."
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
            "\nThe runtime Mermaid file was still "
            "generated successfully."
        )


# ============================================================
# 12. EXECUTE ONE CALCULATION THROUGH LANGGRAPH
# ============================================================

def execute_calculation(
    name: str,
    user_command: str
) -> CalculatorState:
    """
    Execute one operation in the required order:

        Terminal command
            ↓
        Create input state
            ↓
        Invoke LangGraph
            ↓
        Single calculator node
            ↓
        Produce result
            ↓
        Generate runtime graph
    """

    values, operation = parse_user_command(
        user_command
    )

    input_state: CalculatorState = {
        "name": name,
        "values": values,
        "operation": operation
    }

    # --------------------------------------------------------
    # Display graph input
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "LANGGRAPH INPUT"
    )

    print(
        "========================================"
    )

    print("{")

    print(
        f'  "name": "{name}",'
    )

    print(
        f'  "values": {values},'
    )

    print(
        f'  "operation": "{operation}"'
    )

    print("}")

    # --------------------------------------------------------
    # Invoke the actual LangGraph
    # --------------------------------------------------------

    print(
        "\nExecuting LangGraph..."
    )

    final_state = calculator_graph.invoke(
        input_state
    )

    # --------------------------------------------------------
    # Display result before graph generation
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "LANGGRAPH OUTPUT"
    )

    print(
        "========================================"
    )

    print(
        final_state["message"]
    )

    # --------------------------------------------------------
    # Generate runtime-specific graph only now
    # --------------------------------------------------------

    generate_runtime_graph(
        input_state=input_state,
        output_state=final_state
    )

    return final_state


# ============================================================
# 13. DISPLAY HELP
# ============================================================

def display_help() -> None:
    """
    Display supported commands.
    """

    print(
        "\n========================================"
    )

    print(
        "SUPPORTED OPERATIONS"
    )

    print(
        "========================================"
    )

    print(
        "\nAddition:"
    )

    print(
        "  add 9 and 5"
    )

    print(
        "  add 10 20 30"
    )

    print(
        "  sum 100 and 200"
    )

    print(
        "  add -5 and 20"
    )

    print(
        "\nMultiplication:"
    )

    print(
        "  multiply 4 and 6"
    )

    print(
        "  product of 9 and 6"
    )

    print(
        "  multiply 1 2 3 4"
    )

    print(
        "  product 5 5 5"
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

    print(
        "\nThe runtime graph is generated automatically "
        "after each successful operation."
    )


# ============================================================
# 14. MAIN APPLICATION
# ============================================================

def main() -> None:
    """
    Run the interactive LangGraph calculator.
    """

    print(
        "\n========================================"
    )

    print(
        "ASCEND PYTHON LANGGRAPH"
    )

    print(
        "WEEK 4 - TASK 1"
    )

    print(
        "SINGLE NODE RUNTIME GRAPH"
    )

    print(
        "========================================"
    )

    # Save technical compiled graph definition.
    # No PNG is generated or opened at startup.

    save_compiled_graph_definition()

    print(
        "\nThe runtime graph will be created only after "
        "a calculation is successfully executed."
    )

    # --------------------------------------------------------
    # Get user name
    # --------------------------------------------------------

    name = input(
        "\nEnter your name: "
    ).strip()

    if not name:
        name = "User"

    print(
        f"\nHi {name}!"
    )

    display_help()

    # --------------------------------------------------------
    # Interactive calculation loop
    # --------------------------------------------------------

    while True:
        print(
            "\n----------------------------------------"
        )

        user_command = input(
            "What do you want to calculate? "
        ).strip()

        if not user_command:
            continue

        normalized_command = (
            user_command
            .lower()
            .strip()
        )

        # ----------------------------------------------------
        # Exit
        # ----------------------------------------------------

        if normalized_command in [
            "exit",
            "quit",
            "bye"
        ]:
            print(
                "\n========================================"
            )

            print(
                "TASK 1 COMPLETED"
            )

            print(
                "========================================"
            )

            print(
                f"Goodbye {name}!"
            )

            break

        # ----------------------------------------------------
        # Help
        # ----------------------------------------------------

        if normalized_command in [
            "help",
            "show help"
        ]:
            display_help()
            continue

        # ----------------------------------------------------
        # Execute calculation
        # ----------------------------------------------------

        try:
            execute_calculation(
                name=name,
                user_command=user_command
            )

        except ValueError as value_error:
            print(
                "\n========================================"
            )

            print(
                "INVALID INPUT"
            )

            print(
                "========================================"
            )

            print(
                value_error
            )

            print(
                "\nExamples:"
            )

            print(
                "  add 9 and 5"
            )

            print(
                "  product of 9 and 6"
            )

            print(
                "  multiply 2 3 4"
            )

        except Exception as calculation_error:
            print(
                "\n========================================"
            )

            print(
                "CALCULATION FAILED"
            )

            print(
                "========================================"
            )

            print(
                "Error type:",
                type(calculation_error).__name__
            )

            print(
                "Error details:",
                calculation_error
            )


# ============================================================
# 15. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print(
            "\n\nTask 1 stopped by user."
        )

    except Exception as application_error:
        print(
            "\n========================================"
        )

        print(
            "TASK 1 FAILED"
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
            "\nTask 1 terminated."
        )   