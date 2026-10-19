import json
import os
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv

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
from langchain_core.tools import tool

from langchain_openai import ChatOpenAI

from langgraph.graph import (
    END,
    START,
    StateGraph,
)
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# ============================================================
# ASCEND PYTHON LANGGRAPH
# WEEK 4 - TASK 6
# LANGGRAPH + LLM + TOOLS
# MOVIE RECOMMENDATION AGENT
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
    / "task6_runtime_graph.mmd"
)


RUNTIME_GRAPH_PNG_FILE = (
    CURRENT_DIRECTORY
    / "task6_runtime_graph.png"
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
    override=True,
)


# ============================================================
# 3. READ LLM CONFIGURATION
#
# These names match your existing .env convention.
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
# 4. OUTPUT SETTINGS
# ============================================================

MAX_AGENT_ITERATIONS = 10


# ============================================================
# 5. LOCAL MOVIE DATASET
#
# The tool reads this local Python dataset.
# No external movie API is used.
# ============================================================

MOVIE_DATASET = [
    {
        "title": "Interstellar",
        "genre": "Sci-Fi",
        "year": 2014,
        "rating": 8.7,
        "description": (
            "A team of explorers travels through a "
            "wormhole in space in an attempt to find "
            "a new home for humanity."
        ),
        "reason": (
            "Interstellar combines space exploration, "
            "human relationships, scientific ideas, "
            "and emotional depth."
        ),
    },
    {
        "title": "The Martian",
        "genre": "Sci-Fi",
        "year": 2015,
        "rating": 8.0,
        "description": (
            "An astronaut stranded on Mars uses science "
            "and engineering to survive while a rescue "
            "mission is planned."
        ),
        "reason": (
            "The Martian is a practical, optimistic "
            "science-fiction story focused on survival "
            "and problem solving."
        ),
    },
    {
        "title": "Arrival",
        "genre": "Sci-Fi",
        "year": 2016,
        "rating": 7.9,
        "description": (
            "A linguist works with the military to "
            "communicate with visitors from another world."
        ),
        "reason": (
            "Arrival offers thoughtful science fiction, "
            "strong storytelling, and an intelligent "
            "exploration of language and time."
        ),
    },
    {
        "title": "Mad Max: Fury Road",
        "genre": "Action",
        "year": 2015,
        "rating": 8.1,
        "description": (
            "Survivors cross a dangerous wasteland while "
            "escaping a powerful ruler and the ruler's army."
        ),
        "reason": (
            "Mad Max: Fury Road delivers intense action, "
            "strong visual storytelling, and sustained "
            "momentum."
        ),
    },
    {
        "title": "Mission: Impossible - Fallout",
        "genre": "Action",
        "year": 2018,
        "rating": 7.7,
        "description": (
            "An intelligence agent and the agent's team "
            "race to prevent a global catastrophe."
        ),
        "reason": (
            "Mission: Impossible - Fallout offers large "
            "action sequences, suspense, and practical "
            "stunt work."
        ),
    },
    {
        "title": "John Wick",
        "genre": "Action",
        "year": 2014,
        "rating": 7.4,
        "description": (
            "A retired assassin returns to the criminal "
            "underworld after a personal loss."
        ),
        "reason": (
            "John Wick is known for focused action, "
            "stylized choreography, and a fast pace."
        ),
    },
    {
        "title": "The Grand Budapest Hotel",
        "genre": "Comedy",
        "year": 2014,
        "rating": 8.1,
        "description": (
            "A hotel concierge and a lobby worker become "
            "involved in a dispute over a valuable painting."
        ),
        "reason": (
            "The Grand Budapest Hotel combines dry comedy, "
            "distinctive visual design, and eccentric "
            "characters."
        ),
    },
    {
        "title": "Paddington 2",
        "genre": "Comedy",
        "year": 2017,
        "rating": 7.8,
        "description": (
            "Paddington tries to purchase a special gift "
            "and becomes involved in an unexpected mystery."
        ),
        "reason": (
            "Paddington 2 is warm, humorous, family-friendly, "
            "and built around kindness and community."
        ),
    },
    {
        "title": "Jumanji: Welcome to the Jungle",
        "genre": "Comedy",
        "year": 2017,
        "rating": 6.9,
        "description": (
            "A group of students is transported into a "
            "video-game world and must complete the game."
        ),
        "reason": (
            "Jumanji: Welcome to the Jungle mixes adventure, "
            "comedy, teamwork, and accessible entertainment."
        ),
    },
    {
        "title": "The Shawshank Redemption",
        "genre": "Drama",
        "year": 1994,
        "rating": 9.3,
        "description": (
            "Two prisoners develop a long friendship while "
            "dealing with life inside a difficult prison."
        ),
        "reason": (
            "The Shawshank Redemption is a character-driven "
            "drama about endurance, friendship, and hope."
        ),
    },
    {
        "title": "The Pursuit of Happyness",
        "genre": "Drama",
        "year": 2006,
        "rating": 8.0,
        "description": (
            "A struggling parent pursues a career opportunity "
            "while facing severe financial hardship."
        ),
        "reason": (
            "The Pursuit of Happyness is an inspirational "
            "drama centered on persistence and responsibility."
        ),
    },
    {
        "title": "Good Will Hunting",
        "genre": "Drama",
        "year": 1997,
        "rating": 8.3,
        "description": (
            "A mathematically gifted young adult works with "
            "a counselor while deciding how to use that gift."
        ),
        "reason": (
            "Good Will Hunting combines personal growth, "
            "relationships, mentorship, and strong dialogue."
        ),
    },
    {
        "title": "Spider-Man: Into the Spider-Verse",
        "genre": "Animation",
        "year": 2018,
        "rating": 8.4,
        "description": (
            "A student becomes Spider-Man and works with "
            "heroes from several dimensions."
        ),
        "reason": (
            "The movie offers inventive animation, energetic "
            "action, humor, and a strong coming-of-age story."
        ),
    },
    {
        "title": "Coco",
        "genre": "Animation",
        "year": 2017,
        "rating": 8.4,
        "description": (
            "A young musician enters a remarkable world and "
            "learns important truths about family history."
        ),
        "reason": (
            "Coco combines music, colorful animation, family "
            "themes, and an emotionally engaging story."
        ),
    },
    {
        "title": "Inside Out",
        "genre": "Animation",
        "year": 2015,
        "rating": 8.1,
        "description": (
            "Personified feelings help a child adjust to a "
            "major change in life."
        ),
        "reason": (
            "Inside Out uses creative animation to explore "
            "memory, change, family, and emotional development."
        ),
    },
    {
        "title": "Knives Out",
        "genre": "Mystery",
        "year": 2019,
        "rating": 7.9,
        "description": (
            "A detective investigates the death of a wealthy "
            "novelist within a complicated family."
        ),
        "reason": (
            "Knives Out combines mystery, humor, an ensemble "
            "cast, and a carefully structured investigation."
        ),
    },
    {
        "title": "Shutter Island",
        "genre": "Thriller",
        "year": 2010,
        "rating": 8.2,
        "description": (
            "An investigator visits a remote institution "
            "while examining a mysterious disappearance."
        ),
        "reason": (
            "Shutter Island offers psychological tension, "
            "mystery, atmosphere, and layered storytelling."
        ),
    },
    {
        "title": "A Quiet Place",
        "genre": "Thriller",
        "year": 2018,
        "rating": 7.5,
        "description": (
            "A family attempts to survive in silence in a "
            "world threatened by sound-sensitive creatures."
        ),
        "reason": (
            "A Quiet Place creates suspense through sound, "
            "family relationships, and survival challenges."
        ),
    },
]


# ============================================================
# 6. DEFINE LANGGRAPH STATE
# ============================================================

class MovieAgentState(
    TypedDict,
    total=False,
):
    """
    Shared state used by the LangGraph movie agent.

    Input:
        genre
        messages

    Runtime:
        tool_used
        tool_result
        final_answer
        execution_trace
    """

    genre: str

    messages: Annotated[
        list[AnyMessage],
        add_messages,
    ]

    tool_used: bool

    tool_result: str

    final_answer: str

    execution_trace: list[str]


# ============================================================
# 7. VALIDATE CONFIGURATION
# ============================================================

def validate_configuration() -> None:
    """
    Validate the chat-model configuration.
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
        "TASK 6 CONFIGURATION"
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


    print(
        "Local movie records:",
        len(MOVIE_DATASET),
    )


    if missing_configuration:

        raise ValueError(
            "Missing environment variables: "
            + ", ".join(
                missing_configuration
            )
        )


# ============================================================
# 8. NORMALIZE GENRE
# ============================================================

def normalize_genre(
    genre: str,
) -> str:
    """
    Convert common user genre names into dataset names.
    """

    cleaned_genre = (
        genre
        .strip()
        .lower()
    )


    genre_aliases = {
        "sci fi":
            "sci-fi",

        "science fiction":
            "sci-fi",

        "scifi":
            "sci-fi",

        "animated":
            "animation",

        "cartoon":
            "animation",

        "mysteries":
            "mystery",

        "thrillers":
            "thriller",

        "comedies":
            "comedy",

        "dramas":
            "drama",

        "action movies":
            "action",
    }


    return genre_aliases.get(
        cleaned_genre,
        cleaned_genre,
    )


# ============================================================
# 9. LOCAL MOVIE RECOMMENDATION TOOL
# ============================================================

@tool
def movie_recommendation_tool(
    genre: str,
) -> str:
    """
    Search the local movie dataset for movies matching the
    requested genre.

    Use this tool when a user requests a movie recommendation
    based on a genre such as Action, Comedy, Sci-Fi, Drama,
    Animation, Mystery, or Thriller.

    This tool uses only the local dataset and makes no API call.
    """

    normalized_genre = normalize_genre(
        genre
    )


    matching_movies = [
        movie
        for movie in MOVIE_DATASET
        if (
            movie["genre"]
            .lower()
            ==
            normalized_genre
        )
    ]


    available_genres = sorted(
        {
            movie["genre"]
            for movie in MOVIE_DATASET
        }
    )


    if not matching_movies:

        result = {
            "status":
                "not_found",

            "requested_genre":
                genre,

            "available_genres":
                available_genres,

            "movies":
                [],
        }


        return json.dumps(
            result,
            indent=2,
        )


    matching_movies = sorted(
        matching_movies,
        key=lambda movie: movie["rating"],
        reverse=True,
    )


    result = {
        "status":
            "success",

        "requested_genre":
            genre,

        "normalized_genre":
            normalized_genre,

        "available_genres":
            available_genres,

        "movies":
            matching_movies[:3],
    }


    return json.dumps(
        result,
        indent=2,
    )


# ============================================================
# 10. CREATE CHAT MODEL
# ============================================================

def create_chat_model() -> ChatOpenAI:
    """
    Initialize the OpenAI-compatible chat model.
    """

    print(
        "\nInitializing LLM..."
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
        "LLM initialized:"
    )

    print(
        OPENAI_CHAT_MODEL,
    )


    return model


# ============================================================
# 11. RUNTIME LLM COMPONENTS
# ============================================================

base_llm = None

tool_enabled_llm = None


# ============================================================
# 12. LLM SYSTEM PROMPT
# ============================================================

AGENT_SYSTEM_PROMPT = """
You are a movie recommendation agent.

The user supplies a preferred movie genre.

You have access to movie_recommendation_tool, which searches
a local movie dataset.

Rules:

1. Decide whether the user's request requires the movie tool.

2. If the user asks for movie recommendations by genre, you
   must call movie_recommendation_tool with that genre.

3. Do not invent movie records.

4. When a tool result is already present in the conversation,
   do not call the tool again.

5. After receiving the tool result, create a friendly answer.

6. Recommend the highest-rated matching movie as the primary
   recommendation.

7. Briefly mention up to two alternatives when available.

8. Include:
   - recommended movie title
   - genre
   - year
   - rating
   - concise reason

9. If the local dataset has no matching genre, explain that
   clearly and list the genres available in the tool result.

10. Do not claim that an external API or internet search was
    performed. The tool uses a local dataset.
"""


# ============================================================
# 13. NODE 1 - LLM AGENT
# ============================================================

def llm_agent_node(
    state: MovieAgentState,
) -> dict:
    """
    The LLM decides whether it needs the local movie tool.

    First pass:
        Usually requests the tool.

    Second pass:
        Receives the ToolMessage and generates an answer.
    """

    if tool_enabled_llm is None:

        raise RuntimeError(
            "The tool-enabled LLM has not been initialized."
        )


    current_messages = list(
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


    response = tool_enabled_llm.invoke(
        [
            SystemMessage(
                content=AGENT_SYSTEM_PROMPT,
            ),
            *current_messages,
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
            "Use movie recommendation tool."
        )


        for tool_call in tool_calls:

            print(
                "Tool requested:",
                tool_call.get(
                    "name",
                    "Unknown",
                ),
            )


            print(
                "Tool arguments:",
                tool_call.get(
                    "args",
                    {},
                ),
            )


        execution_trace.append(
            "LLM Agent selected Movie Recommendation Tool"
        )


    else:

        print(
            "LLM decision:"
        )

        print(
            "Generate final response without another "
            "tool call."
        )


        execution_trace.append(
            "LLM Agent produced a response"
        )


    return {
        "messages":
            [response],

        "execution_trace":
            execution_trace,
    }


# ============================================================
# 14. CONDITIONAL ROUTER
# ============================================================

def route_after_llm(
    state: MovieAgentState,
) -> Literal[
    "use_tool",
    "final_response",
]:
    """
    Route to ToolNode when the latest AI message contains
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
            "LLM Agent -> Movie Recommendation Tool"
        )


        return "use_tool"


    print(
        "\nRouting:"
    )

    print(
        "LLM Agent -> Final Response"
    )


    return "final_response"


# ============================================================
# 15. TOOLNODE
#
# ToolNode executes the tool selected by the LLM and adds
# the tool result to the messages state.
# ============================================================

movie_tool_node = ToolNode(
    [
        movie_recommendation_tool,
    ]
)


# ============================================================
# 16. TOOL AUDIT NODE
#
# Runs after ToolNode to save the result into state before
# returning to the LLM Agent.
# ============================================================

def capture_tool_result_node(
    state: MovieAgentState,
) -> dict:
    """
    Capture the latest ToolMessage for runtime reporting.
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


    tool_result = ""


    for message in reversed(
        messages
    ):

        if isinstance(
            message,
            ToolMessage,
        ):

            tool_result = str(
                message.content
            )

            break


    print(
        "\n========================================"
    )

    print(
        "MOVIE TOOL EXECUTED"
    )

    print(
        "========================================"
    )


    print(
        "Local dataset result:"
    )

    print(
        tool_result
    )


    execution_trace.append(
        "Movie Recommendation Tool searched local dataset"
    )


    return {
        "tool_used":
            True,

        "tool_result":
            tool_result,

        "execution_trace":
            execution_trace,
    }


# ============================================================
# 17. FINAL RESPONSE NODE
# ============================================================

def final_response_node(
    state: MovieAgentState,
) -> dict:
    """
    Extract the final LLM response from the message history.
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

            if not getattr(
                message,
                "tool_calls",
                [],
            ):

                content = message.content


                if isinstance(
                    content,
                    str,
                ):

                    final_answer = content.strip()


                elif isinstance(
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


                    final_answer = "\n".join(
                        text_parts
                    ).strip()


                else:

                    final_answer = str(
                        content
                    ).strip()


                break


    if not final_answer:

        final_answer = (
            "I could not create a movie recommendation "
            "from the available local dataset."
        )


    execution_trace.append(
        "Final user-friendly movie recommendation generated"
    )


    print(
        "\n========================================"
    )

    print(
        "FINAL MOVIE RECOMMENDATION"
    )

    print(
        "========================================"
    )


    print(
        final_answer
    )


    return {
        "final_answer":
            final_answer,

        "execution_trace":
            execution_trace,
    }


# ============================================================
# 18. BUILD LANGGRAPH
#
#
#                    START
#                      ↓
#                  LLM Agent
#                 ↙         ↘
#          Tool required?    No tool
#               ↓               ↓
#     Movie Recommendation   Final Response
#             Tool              ↓
#               ↓              END
#      Capture Tool Result
#               ↓
#           LLM Agent
#
#
# ============================================================

graph_builder = StateGraph(
    MovieAgentState
)


graph_builder.add_node(
    "llm_agent",
    llm_agent_node,
)


graph_builder.add_node(
    "movie_recommendation_tool",
    movie_tool_node,
)


graph_builder.add_node(
    "capture_tool_result",
    capture_tool_result_node,
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
        "use_tool":
            "movie_recommendation_tool",

        "final_response":
            "final_response",
    },
)


graph_builder.add_edge(
    "movie_recommendation_tool",
    "capture_tool_result",
)


graph_builder.add_edge(
    "capture_tool_result",
    "llm_agent",
)


graph_builder.add_edge(
    "final_response",
    END,
)


movie_agent_graph = graph_builder.compile()


# ============================================================
# 19. CLEAN TEXT FOR MERMAID
# ============================================================

def clean_mermaid_text(
    value: object,
    max_length: int = 220,
) -> str:
    """
    Simplify runtime values for Mermaid labels.
    """

    text_value = str(
        value
    )


    if len(text_value) > max_length:

        text_value = (
            text_value[:max_length]
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
# 20. BUILD RUNTIME MERMAID GRAPH
# ============================================================

def build_runtime_mermaid(
    final_state: MovieAgentState,
) -> str:
    """
    Create a runtime graph showing whether the LLM selected
    the movie tool and the resulting recommendation.
    """

    genre = clean_mermaid_text(
        final_state["genre"]
    )


    tool_used = final_state.get(
        "tool_used",
        False,
    )


    final_answer = clean_mermaid_text(
        final_state["final_answer"]
    )


    if tool_used:

        runtime_mermaid = f"""
flowchart TD

    START([START])

    GENRE["User Genre: {genre}"]

    AGENT["LLM Agent: Decide whether a tool is required"]

    DECISION{{"Use Movie Tool?"}}

    TOOL["Movie Recommendation Tool: Search local dataset"]

    TOOLRESULT["Tool Result Returned"]

    RESPONSE["LLM Response: {final_answer}"]

    END([END])

    START --> GENRE

    GENRE --> AGENT

    AGENT --> DECISION

    DECISION -->|Yes| TOOL

    TOOL --> TOOLRESULT

    TOOLRESULT --> AGENT

    AGENT --> RESPONSE

    RESPONSE --> END
"""

    else:

        runtime_mermaid = f"""
flowchart TD

    START([START])

    GENRE["User Genre: {genre}"]

    AGENT["LLM Agent: Decide whether a tool is required"]

    DECISION{{"Use Movie Tool?"}}

    RESPONSE["LLM Response: {final_answer}"]

    END([END])

    START --> GENRE

    GENRE --> AGENT

    AGENT --> DECISION

    DECISION -->|No| RESPONSE

    RESPONSE --> END
"""


    return runtime_mermaid.strip()


# ============================================================
# 21. GENERATE RUNTIME GRAPH
#
# The graph is generated only after the recommendation
# workflow completes.
# ============================================================

def generate_runtime_graph(
    final_state: MovieAgentState,
) -> None:
    """
    Save runtime Mermaid and attempt to create/open PNG.
    """

    print(
        "\n========================================"
    )

    print(
        "GENERATING TASK 6 RUNTIME GRAPH"
    )

    print(
        "========================================"
    )


    runtime_mermaid = build_runtime_mermaid(
        final_state
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
                    "open it automatically."
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
# 22. EXECUTE ONE MOVIE REQUEST
# ============================================================

def recommend_movie(
    genre: str,
) -> MovieAgentState:
    """
    Execute the complete LangGraph workflow.
    """

    user_request = (
        f"Recommend a {genre} movie using the local "
        "movie recommendation dataset."
    )


    input_state: MovieAgentState = {
        "genre":
            genre,

        "messages":
            [
                HumanMessage(
                    content=user_request,
                )
            ],

        "tool_used":
            False,

        "tool_result":
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
        "User genre:",
        genre,
    )


    print(
        "User request:"
    )

    print(
        user_request,
    )


    final_state = movie_agent_graph.invoke(
        input_state,
        config={
            "recursion_limit":
                MAX_AGENT_ITERATIONS,
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
        "FINAL ANSWER"
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
# 23. DISPLAY HELP
# ============================================================

def display_help() -> None:
    """
    Display supported dataset genres.
    """

    available_genres = sorted(
        {
            movie["genre"]
            for movie in MOVIE_DATASET
        }
    )


    print(
        "\n========================================"
    )

    print(
        "AVAILABLE MOVIE GENRES"
    )

    print(
        "========================================"
    )


    for genre in available_genres:

        print(
            " -",
            genre,
        )


    print(
        "\nExample input:"
    )

    print(
        "Sci-Fi"
    )


    print(
        "\nCommands:"
    )

    print(
        "help"
    )

    print(
        "exit"
    )


# ============================================================
# 24. MAIN APPLICATION
# ============================================================

def main() -> None:
    """
    Initialize Task 6 and start the movie-agent loop.
    """

    global base_llm
    global tool_enabled_llm


    print(
        "\n========================================"
    )

    print(
        "ASCEND PYTHON LANGGRAPH"
    )

    print(
        "WEEK 4 - TASK 6"
    )

    print(
        "LANGGRAPH + LLM + TOOLS"
    )

    print(
        "MOVIE RECOMMENDATION AGENT"
    )

    print(
        "========================================"
    )


    validate_configuration()


    base_llm = create_chat_model()


    # Bind the local movie tool to the LLM.
    # The LLM decides when to request the tool.

    tool_enabled_llm = base_llm.bind_tools(
        [
            movie_recommendation_tool,
        ]
    )


    print(
        "\nMovie recommendation tool bound "
        "to LLM successfully."
    )


    display_help()


    while True:

        print(
            "\n----------------------------------------"
        )


        genre = input(
            "Enter preferred genre: "
        ).strip()


        if not genre:

            continue


        normalized_input = (
            genre
            .lower()
            .strip()
        )


        if normalized_input in [
            "exit",
            "quit",
            "bye",
        ]:

            print(
                "\nTask 6 movie agent stopped."
            )

            break


        if normalized_input in [
            "help",
            "show help",
        ]:

            display_help()

            continue


        try:

            recommend_movie(
                genre
            )


        except Exception as recommendation_error:

            print(
                "\n========================================"
            )

            print(
                "MOVIE RECOMMENDATION FAILED"
            )

            print(
                "========================================"
            )


            print(
                "Error type:",
                type(recommendation_error).__name__,
            )


            print(
                "Error details:",
                recommendation_error,
            )


# ============================================================
# 25. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()


    except KeyboardInterrupt:

        print(
            "\n\nTask 6 stopped by user."
        )


    except Exception as application_error:

        print(
            "\n========================================"
        )

        print(
            "TASK 6 FAILED"
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
            "\nTask 6 terminated due to an unexpected error."
        )