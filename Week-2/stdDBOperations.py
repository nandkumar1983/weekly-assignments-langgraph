import os
from unittest.mock import Base
from dotenv import load_dotenv
from sqlalchemy import Column, Integer, String, create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base


class StudentDBOperations(Base):
    __tablename__ = "students"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    pass
# ...existing code...