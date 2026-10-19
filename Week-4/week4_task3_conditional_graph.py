import os
from pathlib import Path
from typing import Literal, TypedDict

from langchain_core.runnables.graph_mermaid import (
    draw_mermaid_png
)
from langgraph.graph import StateGraph, START, END


# ============================================================
# ASCEND PYTHON LANGGRAPH
# WEEK 4 - TASK 3
# CONDITIONAL GRAPH
# ============================================================


# ============================================================
# 1. FILE PATH CONFIGURATION
# ============================================================

CURRENT_DIRECTORY = Path(
    __file__
).resolve().parent

RUNTIME_GRAPH_MERMAID_FILE = (
    CURRENT_DIRECTORY
    / "task3_runtime_graph.mmd"
)

RUNTIME_GRAPH_PNG_FILE = (
    CURRENT_DIRECTORY
    / "task3_runtime_graph.png"
)


# ============================================================
# 2. DEFINE LANGGRAPH STATE
# ============================================================

class AgentState(TypedDict, total=False):
    """
    Shared state used by the conditional graph.

    First operation:
        number1 operation number2
        result stored in finalNumber

    Second operation:
        number3 operation2 number4
        result stored in finalNumber2

    Runtime route fields:
        route1_taken
        route2_taken
    """

    number1: int
    operation: str
    number2: int

    number3: int
    operation2: str
    number4: int

    finalNumber: int
    finalNumber2: int

    route1_taken: str
    route2_taken: str


# ============================================================
# 3. ROUTER 1 NODE
# ============================================================

def router1(
    state: AgentState
) -> dict:
    """
    Entry node for the first conditional operation.

    The routing decision itself is made by
    route_operation1().
    """

    print(
        "\n========================================"
    )

    print(
        "ROUTER 1 EXECUTED"
    )

    print(
        "========================================"
    )

    print(
        "First expression:",
        state["number1"],
        state["operation"],
        state["number2"]
    )

    return {}


# ============================================================
# 4. ROUTER 1 DECISION FUNCTION
# ============================================================

def route_operation1(
    state: AgentState
) -> Literal["addition", "subtraction"]:
    """
    First conditional router.

    operation = "+" routes to add_node.
    operation = "-" routes to subtract_node.
    """

    operation = state["operation"]

    if operation == "+":

        print(
            "Router 1 decision: add_node"
        )

        return "addition"

    if operation == "-":

        print(
            "Router 1 decision: subtract_node"
        )

        return "subtraction"

    raise ValueError(
        "The first operation must be '+' or '-'."
    )


# ============================================================
# 5. FIRST ADDITION NODE
# ============================================================

def add_node(
    state: AgentState
) -> dict:
    """
    Execute number1 + number2.
    """

    final_number = (
        state["number1"]
        + state["number2"]
    )

    print(
        "\n========================================"
    )

    print(
        "ADD NODE EXECUTED"
    )

    print(
        "========================================"
    )

    print(
        f"{state['number1']} "
        f"+ {state['number2']} "
        f"= {final_number}"
    )

    return {
        "finalNumber":
            final_number,

        "route1_taken":
            "add_node"
    }


# ============================================================
# 6. FIRST SUBTRACTION NODE
# ============================================================

def subtract_node(
    state: AgentState
) -> dict:
    """
    Execute number1 - number2.
    """

    final_number = (
        state["number1"]
        - state["number2"]
    )

    print(
        "\n========================================"
    )

    print(
        "SUBTRACT NODE EXECUTED"
    )

    print(
        "========================================"
    )

    print(
        f"{state['number1']} "
        f"- {state['number2']} "
        f"= {final_number}"
    )

    return {
        "finalNumber":
            final_number,

        "route1_taken":
            "subtract_node"
    }


# ============================================================
# 7. ROUTER 2 NODE
# ============================================================

def router2(
    state: AgentState
) -> dict:
    """
    Entry node for the second conditional operation.

    Both first-operation nodes connect to this node.

    The second routing decision is made by
    route_operation2().
    """

    print(
        "\n========================================"
    )

    print(
        "ROUTER 2 EXECUTED"
    )

    print(
        "========================================"
    )

    print(
        "Second expression:",
        state["number3"],
        state["operation2"],
        state["number4"]
    )

    return {}


# ============================================================
# 8. ROUTER 2 DECISION FUNCTION
# ============================================================

def route_operation2(
    state: AgentState
) -> Literal["addition2", "subtraction2"]:
    """
    Second conditional router.

    operation2 = "+" routes to add_node2.
    operation2 = "-" routes to subtract_node2.
    """

    operation2 = state["operation2"]

    if operation2 == "+":

        print(
            "Router 2 decision: add_node2"
        )

        return "addition2"

    if operation2 == "-":

        print(
            "Router 2 decision: subtract_node2"
        )

        return "subtraction2"

    raise ValueError(
        "The second operation must be '+' or '-'."
    )


# ============================================================
# 9. SECOND ADDITION NODE
# ============================================================

def add_node2(
    state: AgentState
) -> dict:
    """
    Execute number3 + number4.
    """

    final_number2 = (
        state["number3"]
        + state["number4"]
    )

    print(
        "\n========================================"
    )

    print(
        "ADD NODE 2 EXECUTED"
    )

    print(
        "========================================"
    )

    print(
        f"{state['number3']} "
        f"+ {state['number4']} "
        f"= {final_number2}"
    )

    return {
        "finalNumber2":
            final_number2,

        "route2_taken":
            "add_node2"
    }


# ============================================================
# 10. SECOND SUBTRACTION NODE
# ============================================================

def subtract_node2(
    state: AgentState
) -> dict:
    """
    Execute number3 - number4.
    """

    final_number2 = (
        state["number3"]
        - state["number4"]
    )

    print(
        "\n========================================"
    )

    print(
        "SUBTRACT NODE 2 EXECUTED"
    )

    print(
        "========================================"
    )

    print(
        f"{state['number3']} "
        f"- {state['number4']} "
        f"= {final_number2}"
    )

    return {
        "finalNumber2":
            final_number2,

        "route2_taken":
            "subtract_node2"
    }


# ============================================================
# 11. CREATE STATEGRAPH
# ============================================================

graph_builder = StateGraph(
    AgentState
)


# ============================================================
# 12. ADD GRAPH NODES
# ============================================================

graph_builder.add_node(
    "router1",
    router1
)

graph_builder.add_node(
    "add_node",
    add_node
)

graph_builder.add_node(
    "subtract_node",
    subtract_node
)

graph_builder.add_node(
    "router2",
    router2
)

graph_builder.add_node(
    "add_node2",
    add_node2
)

graph_builder.add_node(
    "subtract_node2",
    subtract_node2
)


# ============================================================
# 13. START EDGE
# ============================================================

graph_builder.add_edge(
    START,
    "router1"
)


# ============================================================
# 14. CONDITIONAL EDGE 1
#
# Exactly one conditional edge is configured here.
# ============================================================

graph_builder.add_conditional_edges(
    "router1",
    route_operation1,
    {
        "addition":
            "add_node",

        "subtraction":
            "subtract_node"
    }
)


# ============================================================
# 15. CONNECT FIRST OPERATION TO ROUTER 2
# ============================================================

graph_builder.add_edge(
    "add_node",
    "router2"
)

graph_builder.add_edge(
    "subtract_node",
    "router2"
)


# ============================================================
# 16. CONDITIONAL EDGE 2
#
# Exactly one conditional edge is configured here.
# ============================================================

graph_builder.add_conditional_edges(
    "router2",
    route_operation2,
    {
        "addition2":
            "add_node2",

        "subtraction2":
            "subtract_node2"
    }
)


# ============================================================
# 17. END EDGES
# ============================================================

graph_builder.add_edge(
    "add_node2",
    END
)

graph_builder.add_edge(
    "subtract_node2",
    END
)


# ============================================================
# 18. COMPILE LANGGRAPH
# ============================================================

conditional_graph = (
    graph_builder.compile()
)


# ============================================================
# 19. VALIDATE AN OPERATION
# ============================================================

def validate_operation(
    operation: str,
    operation_name: str
) -> str:
    """
    Validate that an operation is '+' or '-'.
    """

    normalized_operation = (
        operation
        .strip()
    )

    if normalized_operation not in [
        "+",
        "-"
    ]:

        raise ValueError(
            f"{operation_name} must be '+' or '-'."
        )

    return normalized_operation


# ============================================================
# 20. READ INTEGER FROM TERMINAL
# ============================================================

def read_integer(
    prompt: str
) -> int:
    """
    Read and validate one integer from terminal input.
    """

    value_text = input(
        prompt
    ).strip()

    try:

        return int(
            value_text
        )

    except ValueError as error:

        raise ValueError(
            f"'{value_text}' is not a valid integer."
        ) from error


# ============================================================
# 21. ESCAPE TEXT FOR MERMAID
# ============================================================

def escape_mermaid_text(
    value: object
) -> str:
    """
    Escape text used inside Mermaid labels.
    """

    return (
        str(value)
        .replace("&", "&amp;")
        .replace('"', "'")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", " ")
    )


# ============================================================
# 22. BUILD RUNTIME MERMAID GRAPH
# ============================================================

def build_runtime_mermaid(
    input_state: AgentState,
    output_state: AgentState
) -> str:
    """
    Build a runtime-specific graph after LangGraph execution.

    The full conditional design remains visible.

    The selected runtime branches are highlighted.
    Non-selected branches are muted.
    """

    number1 = input_state["number1"]
    operation1 = input_state["operation"]
    number2 = input_state["number2"]

    number3 = input_state["number3"]
    operation2 = input_state["operation2"]
    number4 = input_state["number4"]

    final_number = output_state["finalNumber"]
    final_number2 = output_state["finalNumber2"]

    route1_taken = output_state["route1_taken"]
    route2_taken = output_state["route2_taken"]

    # --------------------------------------------------------
    # First branch styles
    # --------------------------------------------------------

    if route1_taken == "add_node":

        add1_class = "selectedNode"
        subtract1_class = "mutedNode"

        selected_route1 = (
            f"add_node: "
            f"{number1} + {number2} = {final_number}"
        )

    else:

        add1_class = "mutedNode"
        subtract1_class = "selectedNode"

        selected_route1 = (
            f"subtract_node: "
            f"{number1} - {number2} = {final_number}"
        )

    # --------------------------------------------------------
    # Second branch styles
    # --------------------------------------------------------

    if route2_taken == "add_node2":

        add2_class = "selectedNode"
        subtract2_class = "mutedNode"

        selected_route2 = (
            f"add_node2: "
            f"{number3} + {number4} = {final_number2}"
        )

    else:

        add2_class = "mutedNode"
        subtract2_class = "selectedNode"

        selected_route2 = (
            f"subtract_node2: "
            f"{number3} - {number4} = {final_number2}"
        )

    selected_route1 = escape_mermaid_text(
        selected_route1
    )

    selected_route2 = escape_mermaid_text(
        selected_route2
    )

    final_summary = escape_mermaid_text(
        f"finalNumber = {final_number}; "
        f"finalNumber2 = {final_number2}"
    )

    runtime_mermaid = f"""
---
config:
  flowchart:
    curve: linear
---
flowchart TD

    START(["START"])

    ROUTER1["<b>Router 1</b><br/><br/>Operation: {operation1}<br/>Selected route:<br/>{selected_route1}"]

    ADD1["<b>add_node</b><br/><br/>{number1} + {number2}<br/>Result: {number1 + number2}"]

    SUBTRACT1["<b>subtract_node</b><br/><br/>{number1} - {number2}<br/>Result: {number1 - number2}"]

    ROUTER2["<b>Router 2</b><br/><br/>Operation 2: {operation2}<br/>Selected route:<br/>{selected_route2}"]

    ADD2["<b>add_node2</b><br/><br/>{number3} + {number4}<br/>Result: {number3 + number4}"]

    SUBTRACT2["<b>subtract_node2</b><br/><br/>{number3} - {number4}<br/>Result: {number3 - number4}"]

    END(["END<br/><br/>{final_summary}"])

    START --> ROUTER1

    ROUTER1 -->|"+"| ADD1
    ROUTER1 -->|"-"| SUBTRACT1

    ADD1 --> ROUTER2
    SUBTRACT1 --> ROUTER2

    ROUTER2 -->|"+"| ADD2
    ROUTER2 -->|"-"| SUBTRACT2

    ADD2 --> END
    SUBTRACT2 --> END

    classDef startNode fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#1B5E20;
    classDef routerNode fill:#E3F2FD,stroke:#1565C0,stroke-width:2px,color:#0D47A1;
    classDef selectedNode fill:#FFF3CD,stroke:#FF8F00,stroke-width:4px,color:#5D4037;
    classDef mutedNode fill:#F5F5F5,stroke:#BDBDBD,stroke-width:1px,color:#9E9E9E,stroke-dasharray:5 5;
    classDef endNode fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px,color:#4A148C;

    class START startNode;
    class ROUTER1,ROUTER2 routerNode;
    class ADD1 {add1_class};
    class SUBTRACT1 {subtract1_class};
    class ADD2 {add2_class};
    class SUBTRACT2 {subtract2_class};
    class END endNode;
""".strip()

    return runtime_mermaid


# ============================================================
# 23. GENERATE RUNTIME GRAPH
# ============================================================

def generate_runtime_graph(
    input_state: AgentState,
    output_state: AgentState
) -> None:
    """
    Generate the Mermaid and PNG graph after the complete
    conditional LangGraph finishes executing.
    """

    print(
        "\n========================================"
    )

    print(
        "GENERATING TASK 3 RUNTIME GRAPH"
    )

    print(
        "========================================"
    )

    runtime_mermaid = build_runtime_mermaid(
        input_state=input_state,
        output_state=output_state
    )

    # --------------------------------------------------------
    # Save Mermaid graph
    # --------------------------------------------------------

    RUNTIME_GRAPH_MERMAID_FILE.write_text(
        runtime_mermaid,
        encoding="utf-8"
    )

    print(
        "\nRuntime Mermaid graph saved:"
    )

    print(
        RUNTIME_GRAPH_MERMAID_FILE
    )

    # --------------------------------------------------------
    # Generate PNG graph
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
        # Open graph automatically on Windows
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
                    "\nThe PNG was generated but Windows "
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
            "\nThe runtime Mermaid graph was still saved."
        )


# ============================================================
# 24. EXECUTE CONDITIONAL WORKFLOW
# ============================================================

def execute_conditional_workflow(
    number1: int,
    operation1: str,
    number2: int,
    number3: int,
    operation2: str,
    number4: int
) -> AgentState:
    """
    Execute both arithmetic operations through the
    conditional LangGraph.

    Runtime order:

        Input state
            ↓
        router1 conditional edge
            ↓
        add_node or subtract_node
            ↓
        router2 conditional edge
            ↓
        add_node2 or subtract_node2
            ↓
        Final state
            ↓
        Runtime graph generation
    """

    input_state: AgentState = {
        "number1":
            number1,

        "operation":
            operation1,

        "number2":
            number2,

        "number3":
            number3,

        "operation2":
            operation2,

        "number4":
            number4,

        "finalNumber":
            0,

        "finalNumber2":
            0
    }

    # --------------------------------------------------------
    # Display input state
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "LANGGRAPH INPUT STATE"
    )

    print(
        "========================================"
    )

    print("{")

    print(
        f'  "number1": {number1},'
    )

    print(
        f'  "operation": "{operation1}",'
    )

    print(
        f'  "number2": {number2},'
    )

    print(
        f'  "number3": {number3},'
    )

    print(
        f'  "operation2": "{operation2}",'
    )

    print(
        f'  "number4": {number4},'
    )

    print(
        '  "finalNumber": 0,'
    )

    print(
        '  "finalNumber2": 0'
    )

    print("}")

    # --------------------------------------------------------
    # Invoke the actual LangGraph
    # --------------------------------------------------------

    print(
        "\nExecuting conditional LangGraph..."
    )

    final_state = conditional_graph.invoke(
        input_state
    )

    # --------------------------------------------------------
    # Display final state
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "EXPECTED FINAL STATE"
    )

    print(
        "========================================"
    )

    print(
        "AgentState("
    )

    print(
        f"    number1={final_state['number1']},"
    )

    print(
        f'    operation="{final_state["operation"]}",'
    )

    print(
        f"    number2={final_state['number2']},"
    )

    print(
        f"    number3={final_state['number3']},"
    )

    print(
        f"    number4={final_state['number4']},"
    )

    print(
        f'    operation2="{final_state["operation2"]}",'
    )

    print(
        f"    finalNumber={final_state['finalNumber']},"
    )

    print(
        f"    finalNumber2={final_state['finalNumber2']}"
    )

    print(
        ")"
    )

    print(
        "\n========================================"
    )

    print(
        "CALCULATION SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        "First result:"
    )

    print(
        f"{number1} {operation1} {number2} "
        f"= {final_state['finalNumber']}"
    )

    print(
        "\nSecond result:"
    )

    print(
        f"{number3} {operation2} {number4} "
        f"= {final_state['finalNumber2']}"
    )

    print(
        "\nRoutes executed:"
    )

    print(
        "Router 1 ->",
        final_state["route1_taken"]
    )

    print(
        "Router 2 ->",
        final_state["route2_taken"]
    )

    # --------------------------------------------------------
    # Generate runtime graph only after execution
    # --------------------------------------------------------

    generate_runtime_graph(
        input_state=input_state,
        output_state=final_state
    )

    return final_state


# ============================================================
# 25. MAIN APPLICATION
# ============================================================

def main() -> None:
    """
    Run Week 4 Task 3.

    The graph is generated only after the LangGraph
    conditional workflow successfully executes.
    """

    print(
        "\n========================================"
    )

    print(
        "ASCEND PYTHON LANGGRAPH"
    )

    print(
        "WEEK 4 - TASK 3"
    )

    print(
        "CONDITIONAL GRAPH"
    )

    print(
        "========================================"
    )

    print(
        "\nThis graph performs two arithmetic "
        "operations."
    )

    print(
        "Each operation supports '+' or '-'."
    )

    print(
        "\nThe graph will be generated after both "
        "operations execute."
    )

    # --------------------------------------------------------
    # First calculation
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "FIRST OPERATION"
    )

    print(
        "========================================"
    )

    number1 = read_integer(
        "Enter number1: "
    )

    operation1 = validate_operation(
        input(
            "Enter operation1 (+ or -): "
        ),
        "operation1"
    )

    number2 = read_integer(
        "Enter number2: "
    )

    # --------------------------------------------------------
    # Second calculation
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "SECOND OPERATION"
    )

    print(
        "========================================"
    )

    number3 = read_integer(
        "Enter number3: "
    )

    operation2 = validate_operation(
        input(
            "Enter operation2 (+ or -): "
        ),
        "operation2"
    )

    number4 = read_integer(
        "Enter number4: "
    )

    # --------------------------------------------------------
    # Execute graph
    # --------------------------------------------------------

    execute_conditional_workflow(
        number1=number1,
        operation1=operation1,
        number2=number2,
        number3=number3,
        operation2=operation2,
        number4=number4
    )


# ============================================================
# 26. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\n\nTask 3 stopped by user."
        )

    except Exception as application_error:

        print(
            "\n========================================"
        )

        print(
            "TASK 3 FAILED"
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
            "========================================"
        )