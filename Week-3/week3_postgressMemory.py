import json
import os
import re
from datetime import datetime

from dotenv import load_dotenv

from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    delete,
    func,
    select,
    text
)

from sqlalchemy.engine import URL
from sqlalchemy.orm import declarative_base, sessionmaker

from langchain_core.messages import HumanMessage, SystemMessage
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

openai_base_url = os.getenv("OPENAI_BASE_URL")
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_chat_model = os.getenv("OPENAI_CHAT_MODEL")


# ==========================================================
# 3. READ POSTGRESQL CONFIGURATION
# ==========================================================

db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT", "5432")
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")


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

print("OpenAI Base URL   :", openai_base_url)
print(
    "OpenAI API Key    :",
    "Loaded" if openai_api_key else "Missing"
)
print("OpenAI Chat Model :", openai_chat_model)

print("Database Host     :", db_host)
print("Database Port     :", db_port)
print("Database Name     :", db_name)
print("Database User     :", db_user)
print(
    "Database Password :",
    "Loaded" if db_password else "Missing"
)

print("=========================================")

if missing_variables:
    raise ValueError(
        "Missing environment variables: "
        + ", ".join(missing_variables)
    )


# ==========================================================
# 5. CREATE DATABASE CONNECTION
# ==========================================================

database_url = URL.create(
    drivername="postgresql+psycopg2",
    username=db_user,
    password=db_password,
    host=db_host,
    port=int(db_port),
    database=db_name
)

engine = create_engine(
    database_url,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False
)

Base = declarative_base()


# ==========================================================
# 6. USER PREFERENCE TABLE
# ==========================================================

class UserPreference(Base):

    __tablename__ = "user_preferences"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    user_id = Column(
        String(100),
        nullable=False,
        index=True
    )

    preference_type = Column(
        String(100),
        nullable=False,
        index=True
    )

    preference_value = Column(
        Text,
        nullable=False
    )

    created_at = Column(
        DateTime,
        nullable=False,
        default=datetime.now
    )

    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.now,
        onupdate=datetime.now
    )

    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "preference_type",
            "preference_value",
            name="uq_user_preference_value"
        ),
    )


# ==========================================================
# 7. CHAT HISTORY TABLE
# ==========================================================

class ChatHistory(Base):

    __tablename__ = "chat_history"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    user_id = Column(
        String(100),
        nullable=False,
        index=True
    )

    role = Column(
        String(20),
        nullable=False
    )

    message = Column(
        Text,
        nullable=False
    )

    created_at = Column(
        DateTime,
        nullable=False,
        default=datetime.now
    )


# ==========================================================
# 8. CREATE TABLES
# ==========================================================

Base.metadata.create_all(bind=engine)


# ==========================================================
# 9. FIX OLD UNIQUE CONSTRAINT AUTOMATICALLY
# ==========================================================

def migrate_database_constraints():

    with engine.begin() as connection:

        # Remove the old constraint that permits only
        # one value per user and preference type.
        connection.execute(
            text(
                """
                ALTER TABLE public.user_preferences
                DROP CONSTRAINT IF EXISTS
                uq_user_preference
                """
            )
        )

        # Check whether the required three-column
        # constraint already exists.
        constraint_exists = connection.execute(
            text(
                """
                SELECT 1
                FROM pg_constraint
                WHERE conname =
                    'uq_user_preference_value'
                """
            )
        ).scalar()

        if not constraint_exists:

            connection.execute(
                text(
                    """
                    ALTER TABLE public.user_preferences
                    ADD CONSTRAINT
                    uq_user_preference_value
                    UNIQUE (
                        user_id,
                        preference_type,
                        preference_value
                    )
                    """
                )
            )


try:

    migrate_database_constraints()

    with engine.connect() as connection:
        connection.execute(select(1))

    print("\nPostgreSQL connection: Successful")
    print("Database tables: Verified or created")
    print("Database constraint: Verified or migrated")

except Exception as error:

    print("\nPostgreSQL initialization failed.")
    print(type(error).__name__)
    print(error)

    raise


# ==========================================================
# 10. INITIALIZE LLM
# ==========================================================

llm = ChatOpenAI(
    model=openai_chat_model,
    base_url=openai_base_url,
    api_key=openai_api_key
)


# ==========================================================
# 11. GET TEXT FROM LLM RESPONSE
# ==========================================================

def get_response_text(response):

    content = response.content

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
                    text_parts.append(str(text_value))

        return "\n".join(text_parts).strip()

    return str(content).strip()


# ==========================================================
# 12. NORMALIZE PREFERENCE TYPE
# ==========================================================

def normalize_preference_type(preference_type):

    if not preference_type:
        return None

    normalized = (
        str(preference_type)
        .lower()
        .strip()
        .replace(" ", "_")
    )

    aliases = {
        "food": "favorite_food",
        "foods": "favorite_food",
        "favorite_food": "favorite_food",
        "favourite_food": "favorite_food",

        "drink": "favorite_drink",
        "drinks": "favorite_drink",
        "beverage": "favorite_drink",
        "beverages": "favorite_drink",
        "favorite_drink": "favorite_drink",
        "favourite_drink": "favorite_drink",

        "city": "favorite_city",
        "favorite_city": "favorite_city",
        "favourite_city": "favorite_city",

        "hobby": "hobby",
        "hobbies": "hobby",

        "movie": "favorite_movie",
        "favorite_movie": "favorite_movie",
        "favourite_movie": "favorite_movie",

        "language": "programming_language",
        "preferred_language": "programming_language",
        "programming_language": "programming_language",
        "favorite_programming_language":
            "programming_language",

        "sport": "favorite_sport",
        "favorite_sport": "favorite_sport",
        "favourite_sport": "favorite_sport",

        "destination": "travel_destination",
        "travel_destination": "travel_destination",
        "favorite_travel_destination":
            "travel_destination",

        "book": "favorite_book",
        "favorite_book": "favorite_book",
        "favourite_book": "favorite_book",

        "music": "favorite_music",
        "favorite_music": "favorite_music",
        "favourite_music": "favorite_music"
    }

    return aliases.get(
        normalized,
        normalized
    )


# ==========================================================
# 13. DISPLAY-FRIENDLY PREFERENCE NAME
# ==========================================================

def get_display_name(preference_type):

    display_names = {
        "favorite_food": "favorite food",
        "favorite_drink": "favorite drink",
        "favorite_city": "favorite city",
        "hobby": "hobby",
        "favorite_movie": "favorite movie",
        "programming_language":
            "preferred programming language",
        "favorite_sport": "favorite sport",
        "travel_destination":
            "favorite travel destination",
        "favorite_book": "favorite book",
        "favorite_music": "favorite music"
    }

    return display_names.get(
        preference_type,
        preference_type.replace("_", " ")
    )


# ==========================================================
# 14. SAVE CHAT MESSAGE
# ==========================================================

def save_chat_message(user_id, role, message):

    with SessionLocal() as database:

        chat_record = ChatHistory(
            user_id=user_id,
            role=role,
            message=message,
            created_at=datetime.now()
        )

        database.add(chat_record)
        database.commit()


# ==========================================================
# 15. ADD PREFERENCE
# ==========================================================

def add_preference(
    user_id,
    preference_type,
    preference_value
):

    preference_type = normalize_preference_type(
        preference_type
    )

    preference_value = str(
        preference_value
    ).strip()

    with SessionLocal() as database:

        existing_record = database.execute(
            select(UserPreference).where(
                UserPreference.user_id == user_id,
                UserPreference.preference_type
                == preference_type,
                func.lower(
                    UserPreference.preference_value
                )
                == preference_value.lower()
            )
        ).scalar_one_or_none()

        if existing_record:
            return False

        new_preference = UserPreference(
            user_id=user_id,
            preference_type=preference_type,
            preference_value=preference_value,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        database.add(new_preference)
        database.commit()

        return True


# ==========================================================
# 16. UPDATE SPECIFIC PREFERENCE
# ==========================================================

def update_preference(
    user_id,
    preference_type,
    old_value,
    new_value
):

    preference_type = normalize_preference_type(
        preference_type
    )

    old_value = str(old_value).strip()
    new_value = str(new_value).strip()

    with SessionLocal() as database:

        existing_record = database.execute(
            select(UserPreference).where(
                UserPreference.user_id == user_id,
                UserPreference.preference_type
                == preference_type,
                func.lower(
                    UserPreference.preference_value
                )
                == old_value.lower()
            )
        ).scalar_one_or_none()

        if not existing_record:
            return False

        duplicate_new_value = database.execute(
            select(UserPreference).where(
                UserPreference.user_id == user_id,
                UserPreference.preference_type
                == preference_type,
                func.lower(
                    UserPreference.preference_value
                )
                == new_value.lower()
            )
        ).scalar_one_or_none()

        if (
            duplicate_new_value
            and duplicate_new_value.id
            != existing_record.id
        ):

            database.delete(existing_record)
            database.commit()

            return True

        existing_record.preference_value = new_value
        existing_record.updated_at = datetime.now()

        database.commit()

        return True


# ==========================================================
# 17. REPLACE ALL VALUES OF ONE CATEGORY
# ==========================================================

def replace_preference_type(
    user_id,
    preference_type,
    new_value
):

    preference_type = normalize_preference_type(
        preference_type
    )

    previous_values = get_preferences_by_type(
        user_id,
        preference_type
    )

    with SessionLocal() as database:

        database.execute(
            delete(UserPreference).where(
                UserPreference.user_id == user_id,
                UserPreference.preference_type
                == preference_type
            )
        )

        new_record = UserPreference(
            user_id=user_id,
            preference_type=preference_type,
            preference_value=str(new_value).strip(),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        database.add(new_record)
        database.commit()

    return previous_values


# ==========================================================
# 18. GET PREFERENCES BY TYPE
# ==========================================================

def get_preferences_by_type(
    user_id,
    preference_type
):

    preference_type = normalize_preference_type(
        preference_type
    )

    with SessionLocal() as database:

        records = database.execute(
            select(UserPreference)
            .where(
                UserPreference.user_id == user_id,
                UserPreference.preference_type
                == preference_type
            )
            .order_by(UserPreference.id)
        ).scalars().all()

        return [
            record.preference_value
            for record in records
        ]


# ==========================================================
# 19. GET ALL PREFERENCES
# ==========================================================

def get_all_preferences(user_id):

    with SessionLocal() as database:

        records = database.execute(
            select(UserPreference)
            .where(
                UserPreference.user_id == user_id
            )
            .order_by(
                UserPreference.preference_type,
                UserPreference.id
            )
        ).scalars().all()

        preferences = {}

        for record in records:

            if record.preference_type not in preferences:
                preferences[record.preference_type] = []

            preferences[
                record.preference_type
            ].append(
                record.preference_value
            )

        return preferences


# ==========================================================
# 20. DISPLAY PREFERENCES
# ==========================================================

def display_preferences(user_id):

    preferences = get_all_preferences(user_id)

    print("\n========== STORED PREFERENCES ==========")

    if not preferences:

        print(
            "No preferences are stored for this user."
        )

    else:

        for preference_type, values in (
            preferences.items()
        ):

            display_name = get_display_name(
                preference_type
            )

            print(f"\n{display_name.title()}:")

            for index, value in enumerate(
                values,
                start=1
            ):

                print(f"  {index}. {value}")

    print("========================================")


# ==========================================================
# 21. DISPLAY CHAT HISTORY
# ==========================================================

def display_chat_history(user_id):

    with SessionLocal() as database:

        messages = database.execute(
            select(ChatHistory)
            .where(
                ChatHistory.user_id == user_id
            )
            .order_by(ChatHistory.id)
        ).scalars().all()

    print("\n========== PERSISTENT CHAT HISTORY ==========")

    if not messages:

        print(
            "No chat history is stored for this user."
        )

    else:

        for message in messages:

            if message.role == "user":
                print("User      :", message.message)

            else:
                print("Assistant :", message.message)

    print("=============================================")


# ==========================================================
# 22. CLEAR USER MEMORY
# ==========================================================

def clear_user_memory(user_id):

    with SessionLocal() as database:

        database.execute(
            delete(ChatHistory).where(
                ChatHistory.user_id == user_id
            )
        )

        database.execute(
            delete(UserPreference).where(
                UserPreference.user_id == user_id
            )
        )

        database.commit()


# ==========================================================
# 23. REGEX MESSAGE ANALYSIS
# ==========================================================

def analyze_with_regex(user_message):

    message = user_message.strip()

    # ------------------------------------------------------
    # UPDATE A SPECIFIC DRINK
    # ------------------------------------------------------

    drink_update_patterns = [
        r"^change drink (.+?) to (.+?)[.!]?$",
        r"^replace drink (.+?) with (.+?)[.!]?$",
        r"^update drink (.+?) to (.+?)[.!]?$"
    ]

    for pattern in drink_update_patterns:

        match = re.match(
            pattern,
            message,
            flags=re.IGNORECASE
        )

        if match:

            return {
                "action": "UPDATE",
                "preference_type": "favorite_drink",
                "old_value": match.group(1).strip(),
                "new_value": match.group(2).strip()
            }

    # ------------------------------------------------------
    # UPDATE A SPECIFIC FOOD
    # ------------------------------------------------------

    food_update_patterns = [
        r"^change food (.+?) to (.+?)[.!]?$",
        r"^replace food (.+?) with (.+?)[.!]?$",
        r"^update food (.+?) to (.+?)[.!]?$"
    ]

    for pattern in food_update_patterns:

        match = re.match(
            pattern,
            message,
            flags=re.IGNORECASE
        )

        if match:

            return {
                "action": "UPDATE",
                "preference_type": "favorite_food",
                "old_value": match.group(1).strip(),
                "new_value": match.group(2).strip()
            }

    # ------------------------------------------------------
    # GENERIC UPDATE
    # ------------------------------------------------------

    generic_update_patterns = [
        r"^change (.+?) to (.+?)[.!]?$",
        r"^replace (.+?) with (.+?)[.!]?$",
        r"^update (.+?) to (.+?)[.!]?$"
    ]

    for pattern in generic_update_patterns:

        match = re.match(
            pattern,
            message,
            flags=re.IGNORECASE
        )

        if match:

            # Let the LLM classify generic updates.
            return None

    # ------------------------------------------------------
    # REPLACE DRINK CATEGORY
    # ------------------------------------------------------

    drink_replace_patterns = [
        r"^change my favorite drink to (.+?)[.!]?$",
        r"^change my favourite drink to (.+?)[.!]?$",
        r"^update my favorite drink to (.+?)[.!]?$"
    ]

    for pattern in drink_replace_patterns:

        match = re.match(
            pattern,
            message,
            flags=re.IGNORECASE
        )

        if match:

            return {
                "action": "REPLACE",
                "preference_type": "favorite_drink",
                "old_value": None,
                "new_value": match.group(1).strip()
            }

    # ------------------------------------------------------
    # REPLACE FOOD CATEGORY
    # ------------------------------------------------------

    food_replace_patterns = [
        r"^change my favorite food to (.+?)[.!]?$",
        r"^change my favourite food to (.+?)[.!]?$",
        r"^update my favorite food to (.+?)[.!]?$"
    ]

    for pattern in food_replace_patterns:

        match = re.match(
            pattern,
            message,
            flags=re.IGNORECASE
        )

        if match:

            return {
                "action": "REPLACE",
                "preference_type": "favorite_food",
                "old_value": None,
                "new_value": match.group(1).strip()
            }

    # ------------------------------------------------------
    # ADD DRINK
    # ------------------------------------------------------

    drink_patterns = [
        r"^my favorite drink is (.+?)[.!]?$",
        r"^my favourite drink is (.+?)[.!]?$",
        r"^my preferred drink is (.+?)[.!]?$",
        r"^my favorite beverage is (.+?)[.!]?$",
        r"^my favourite beverage is (.+?)[.!]?$",
        r"^i love the drink as (.+?)[.!]?$",
        r"^i love the drink (.+?)[.!]?$",
        r"^i like the drink as (.+?)[.!]?$",
        r"^i like the drink (.+?)[.!]?$",
        r"^i love drinking (.+?)[.!]?$",
        r"^i like drinking (.+?)[.!]?$",
        r"^i also like drinking (.+?)[.!]?$",
        r"^add (.+?) as my favorite drink[.!]?$",
        r"^add (.+?) as another favorite drink[.!]?$",
        r"^add one more favorite drink as (.+?)[.!]?$",
        r"^i would like to add another favorite drink option as (.+?)[.!]?$"
    ]

    for pattern in drink_patterns:

        match = re.match(
            pattern,
            message,
            flags=re.IGNORECASE
        )

        if match:

            return {
                "action": "ADD",
                "preference_type": "favorite_drink",
                "old_value": None,
                "new_value": match.group(1).strip()
            }

    # ------------------------------------------------------
    # ADD FOOD
    # ------------------------------------------------------

    food_patterns = [
        r"^my favorite food is (.+?)[.!]?$",
        r"^my favourite food is (.+?)[.!]?$",
        r"^my preferred food is (.+?)[.!]?$",
        r"^i love eating (.+?)[.!]?$",
        r"^i like eating (.+?)[.!]?$",
        r"^in the non veg i like (.+?)[.!]?$",
        r"^in non veg i like (.+?)[.!]?$",
        r"^add one more favorite food as (.+?)[.!]?$",
        r"^add one more favourite food as (.+?)[.!]?$",
        r"^add (.+?) as another favorite food[.!]?$",
        r"^i would like to add another favorite food option as (.+?)[.!]?$"
    ]

    for pattern in food_patterns:

        match = re.match(
            pattern,
            message,
            flags=re.IGNORECASE
        )

        if match:

            return {
                "action": "ADD",
                "preference_type": "favorite_food",
                "old_value": None,
                "new_value": match.group(1).strip()
            }

    # ------------------------------------------------------
    # OTHER EXPLICIT PREFERENCES
    # ------------------------------------------------------

    other_patterns = [
        (
            r"^my favorite city is (.+?)[.!]?$",
            "favorite_city"
        ),
        (
            r"^my favourite city is (.+?)[.!]?$",
            "favorite_city"
        ),
        (
            r"^my hobby is (.+?)[.!]?$",
            "hobby"
        ),
        (
            r"^my hobbies are (.+?)[.!]?$",
            "hobby"
        ),
        (
            r"^my favorite movie is (.+?)[.!]?$",
            "favorite_movie"
        ),
        (
            r"^my favourite movie is (.+?)[.!]?$",
            "favorite_movie"
        ),
        (
            r"^my preferred programming language is (.+?)[.!]?$",
            "programming_language"
        ),
        (
            r"^my favorite programming language is (.+?)[.!]?$",
            "programming_language"
        ),
        (
            r"^my favorite sport is (.+?)[.!]?$",
            "favorite_sport"
        ),
        (
            r"^my favorite book is (.+?)[.!]?$",
            "favorite_book"
        ),
        (
            r"^my favorite music is (.+?)[.!]?$",
            "favorite_music"
        )
    ]

    for pattern, preference_type in other_patterns:

        match = re.match(
            pattern,
            message,
            flags=re.IGNORECASE
        )

        if match:

            return {
                "action": "ADD",
                "preference_type": preference_type,
                "old_value": None,
                "new_value": match.group(1).strip()
            }

    return None


# ==========================================================
# 24. LLM MESSAGE ANALYSIS
# ==========================================================

def analyze_with_llm(user_message):

    analysis_prompt = f"""
Analyze the user's message for a PostgreSQL personal
preference application.

User message:

{user_message}

Supported preference types:

favorite_food
favorite_drink
favorite_city
hobby
favorite_movie
programming_language
favorite_sport
travel_destination
favorite_book
favorite_music

Classification rules:

1. Food dishes, cuisines and edible items must use
   favorite_food.

2. Drinks, beverages and drink brands must use
   favorite_drink.

3. Food examples:
   Butter Chicken
   Mexican Tacos
   Jeera Rice
   Pizza
   South Indian food
   Mac and Cheese
   Chicken Tandoori

4. Drink examples:
   Coffee
   Tea
   Orange Juice
   Coca-Cola
   Kingfisher Ultra
   Mineral Water

5. Never classify a drink as favorite_food.

6. Never classify a food as favorite_drink.

7. If the user says "I love Butter Chicken", classify
   Butter Chicken as favorite_food.

8. If the user says "I love coffee", classify coffee
   as favorite_drink.

9. If the message is ambiguous, return NONE instead
   of guessing.

Supported actions:

ADD:
The user is adding another preference.

UPDATE:
The user identifies an existing value and its replacement.

REPLACE:
The user wants all values in a preference category
replaced by one new value.

QUERY:
The user asks for stored preferences.

NONE:
No database preference operation is requested.

Return only valid JSON.

ADD example:

{{
    "action": "ADD",
    "preference_type": "favorite_drink",
    "old_value": null,
    "new_value": "Kingfisher Ultra"
}}

UPDATE example:

{{
    "action": "UPDATE",
    "preference_type": "favorite_food",
    "old_value": "Butter Chicken",
    "new_value": "Mexican Tacos"
}}

QUERY example:

{{
    "action": "QUERY",
    "preference_type": "favorite_drink",
    "old_value": null,
    "new_value": null
}}

NONE example:

{{
    "action": "NONE",
    "preference_type": null,
    "old_value": null,
    "new_value": null
}}

Do not return Markdown.
Do not provide explanations.
"""

    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "Classify personal preference database "
                    "operations and return JSON only."
                )
            ),
            HumanMessage(content=analysis_prompt)
        ]
    )

    raw_result = get_response_text(response)

    raw_result = (
        raw_result
        .replace("```json", "")
        .replace("```JSON", "")
        .replace("```", "")
        .strip()
    )

    json_start = raw_result.find("{")
    json_end = raw_result.rfind("}")

    if json_start >= 0 and json_end > json_start:

        raw_result = raw_result[
            json_start:json_end + 1
        ]

    result = json.loads(raw_result)

    result["action"] = str(
        result.get("action", "NONE")
    ).upper()

    result["preference_type"] = (
        normalize_preference_type(
            result.get("preference_type")
        )
    )

    return result


# ==========================================================
# 25. COMBINED ANALYSIS
# ==========================================================

def analyze_message(user_message):

    regex_result = analyze_with_regex(
        user_message
    )

    if regex_result:
        return regex_result

    try:

        return analyze_with_llm(
            user_message
        )

    except Exception as error:

        print(
            "[WARNING] Message analysis failed:",
            type(error).__name__,
            error
        )

        return {
            "action": "NONE",
            "preference_type": None,
            "old_value": None,
            "new_value": None
        }


# ==========================================================
# 26. PROCESS ADD
# ==========================================================

def process_add(
    user_id,
    preference_type,
    new_value
):

    supported_types = {
        "favorite_food",
        "favorite_drink",
        "favorite_city",
        "hobby",
        "favorite_movie",
        "programming_language",
        "favorite_sport",
        "travel_destination",
        "favorite_book",
        "favorite_music"
    }

    preference_type = normalize_preference_type(
        preference_type
    )

    if not preference_type or not new_value:

        return (
            "I could not identify the preference type "
            "or value. Use a complete statement such as "
            "'My favorite drink is coffee'."
        )

    if preference_type not in supported_types:

        return (
            f"I identified {new_value}, but could not "
            "determine a supported preference category."
        )

    added = add_preference(
        user_id,
        preference_type,
        new_value
    )

    display_name = get_display_name(
        preference_type
    )

    if added:

        return (
            f"Got it! I added {new_value} to your "
            f"{display_name} preferences."
        )

    return (
        f"{new_value} is already stored in your "
        f"{display_name} preferences."
    )


# ==========================================================
# 27. PROCESS UPDATE
# ==========================================================

def process_update(
    user_id,
    preference_type,
    old_value,
    new_value
):

    if (
        not preference_type
        or not old_value
        or not new_value
    ):

        return (
            "Please specify the preference category, "
            "old value and new value."
        )

    updated = update_preference(
        user_id,
        preference_type,
        old_value,
        new_value
    )

    display_name = get_display_name(
        preference_type
    )

    if updated:

        return (
            f"Updated successfully. Your previous "
            f"{display_name} value {old_value} was "
            f"changed to {new_value}."
        )

    return (
        f"I could not find {old_value} in your stored "
        f"{display_name} preferences."
    )


# ==========================================================
# 28. PROCESS REPLACE
# ==========================================================

def process_replace(
    user_id,
    preference_type,
    new_value
):

    if not preference_type or not new_value:

        return (
            "I could not identify the preference "
            "category or new value."
        )

    previous_values = replace_preference_type(
        user_id,
        preference_type,
        new_value
    )

    display_name = get_display_name(
        preference_type
    )

    if previous_values:

        return (
            f"Your previous {display_name} preferences "
            f"were {', '.join(previous_values)}. "
            f"I replaced them with {new_value}."
        )

    return (
        f"Got it! Your {display_name} is now "
        f"{new_value}."
    )


# ==========================================================
# 29. PROCESS QUERY
# ==========================================================

def process_query(
    user_id,
    preference_type
):

    if not preference_type:

        return (
            "Please specify which preference category "
            "you want me to retrieve."
        )

    values = get_preferences_by_type(
        user_id,
        preference_type
    )

    display_name = get_display_name(
        preference_type
    )

    if not values:

        return (
            f"You have not shared any "
            f"{display_name} preferences yet."
        )

    result_lines = [
        f"{index}. {value}"
        for index, value in enumerate(
            values,
            start=1
        )
    ]

    return (
        f"Your stored {display_name} preferences are:\n"
        + "\n".join(result_lines)
    )


# ==========================================================
# 30. GENERAL PERSONALIZED RESPONSE
# ==========================================================

def generate_personalized_response(
    user_id,
    user_message
):

    preferences = get_all_preferences(user_id)

    if preferences:

        context_lines = []

        for preference_type, values in (
            preferences.items()
        ):

            context_lines.append(
                f"- {get_display_name(preference_type)}: "
                f"{', '.join(values)}"
            )

        preference_context = "\n".join(
            context_lines
        )

    else:

        preference_context = (
            "No stored preferences are available."
        )

    system_prompt = f"""
You are a friendly personalized assistant.

The following preferences were retrieved from PostgreSQL:

{preference_context}

Use these preferences only when relevant.

Food and drink are separate categories.

Do not claim that information was stored or updated.
Only the Python application changes the database.

Do not invent preferences.

Keep the response concise.
"""

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
    )

    return get_response_text(response)


# ==========================================================
# 31. MAIN APPLICATION
# ==========================================================

def main():

    print("\n================================================")
    print(" PERSONALIZED CHAT ASSISTANT - POSTGRESQL MEMORY")
    print("================================================")

    print("\nOpenAI model:", openai_chat_model)

    print("\nCommands:")
    print("  preferences - Show stored preferences")
    print("  history     - Show stored chat history")
    print("  clear       - Delete your stored memory")
    print("  exit        - Exit the application")

    print("\nExample inputs:")
    print("  My favorite food is Butter Chicken")
    print("  I love Mexican Tacos")
    print("  My favorite drink is coffee")
    print("  I love the drink as Kingfisher Ultra")
    print("  What foods do I like?")
    print("  What drinks do I like?")
    print("  Change food Butter Chicken to Pizza")
    print("  Change drink coffee to orange juice")

    user_id = input(
        "\nEnter User ID [default: njadhav]: "
    ).strip()

    if not user_id:
        user_id = "njadhav"

    print(f"\nWelcome, {user_id}!")

    display_preferences(user_id)

    while True:

        print("\n----------------------------------------------")

        user_input = input("You: ").strip()

        if not user_input:

            print(
                "Assistant: Please enter a message."
            )

            continue

        normalized_input = user_input.lower().strip()

        # --------------------------------------------------
        # EXIT
        # --------------------------------------------------

        if normalized_input in [
            "exit",
            "quit",
            "bye"
        ]:

            assistant_response = (
                "Goodbye! Your preferences remain "
                "stored in PostgreSQL."
            )

            save_chat_message(
                user_id,
                "user",
                user_input
            )

            save_chat_message(
                user_id,
                "assistant",
                assistant_response
            )

            print(
                "\nAssistant:",
                assistant_response
            )

            break

        # --------------------------------------------------
        # DISPLAY PREFERENCES
        # --------------------------------------------------

        if normalized_input in [
            "preferences",
            "show preferences",
            "show my preferences"
        ]:

            display_preferences(user_id)
            continue

        # --------------------------------------------------
        # DISPLAY HISTORY
        # --------------------------------------------------

        if normalized_input in [
            "history",
            "show history",
            "show chat history"
        ]:

            display_chat_history(user_id)
            continue

        # --------------------------------------------------
        # CLEAR MEMORY
        # --------------------------------------------------

        if normalized_input == "clear":

            clear_user_memory(user_id)

            print(
                "\nAssistant: Your stored preferences "
                "and chat history were deleted."
            )

            continue

        # --------------------------------------------------
        # SAVE USER MESSAGE
        # --------------------------------------------------

        save_chat_message(
            user_id,
            "user",
            user_input
        )

        # --------------------------------------------------
        # ANALYZE MESSAGE
        # --------------------------------------------------

        preference_data = analyze_message(
            user_input
        )

        action = preference_data.get(
            "action",
            "NONE"
        )

        preference_type = normalize_preference_type(
            preference_data.get(
                "preference_type"
            )
        )

        old_value = preference_data.get(
            "old_value"
        )

        new_value = preference_data.get(
            "new_value"
        )

        print("\n[DEBUG] Action:", action)

        print(
            "[DEBUG] Preference type:",
            preference_type
        )

        print(
            "[DEBUG] Old value:",
            old_value
        )

        print(
            "[DEBUG] New value:",
            new_value
        )

        # --------------------------------------------------
        # EXECUTE OPERATION
        # --------------------------------------------------

        try:

            if action == "ADD":

                assistant_response = process_add(
                    user_id,
                    preference_type,
                    new_value
                )

            elif action == "UPDATE":

                assistant_response = process_update(
                    user_id,
                    preference_type,
                    old_value,
                    new_value
                )

            elif action == "REPLACE":

                assistant_response = process_replace(
                    user_id,
                    preference_type,
                    new_value
                )

            elif action == "QUERY":

                assistant_response = process_query(
                    user_id,
                    preference_type
                )

            else:

                assistant_response = (
                    generate_personalized_response(
                        user_id,
                        user_input
                    )
                )

            save_chat_message(
                user_id,
                "assistant",
                assistant_response
            )

            print(
                "\nAssistant:",
                assistant_response
            )

        except Exception as error:

            database_error_message = (
                "The operation could not be completed."
            )

            save_chat_message(
                user_id,
                "assistant",
                database_error_message
            )

            print("\nOperation failed:")
            print(type(error).__name__)
            print(error)


# ==========================================================
# 32. EXECUTE APPLICATION
# ==========================================================

if __name__ == "__main__":
    main()