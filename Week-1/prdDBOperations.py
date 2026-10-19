import os
from dotenv import load_dotenv
from sqlalchemy import Column, Float, Integer, String, create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

#Load environment variables from .env file
load_dotenv()

#Load Database connection details from environment variables
db_user = os.getenv("PG_USER")
db_password = os.getenv("PG_PASSWORD")
db_host = os.getenv("PG_HOST")
db_port = os.getenv("PG_PORT")

#Create a connection string for PostgreSQL
connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/postgres"

#Create a SQLAlchemy engine to connect to the PostgreSQL database
engine = create_engine(connection_string, echo=True)

#Session object to manage database transactions
Session = sessionmaker(bind=engine)

#Create a base class for declarative class definitions
Base = declarative_base()

class Product(Base):
    __tablename__ = "Product"
    #product_id should be primary key and generate automatically
    product_id = Column("product_id", Integer, primary_key=True, autoincrement=True)
    product_name = Column(String(50), nullable=False)
    product_price = Column(Float, nullable=True)
    product_quantity = Column(Integer, nullable=True)

    #Create table
    def create_table(self):
        Base.metadata.create_all(engine)

    #Insert product record into the product table
    def insert_product(self, product_name:str, product_price:int=None, product_quantity:int=None):
        #Get session object using factory method
        session = Session()
        try:
            new_product = Product(product_name=product_name, product_price=product_price, product_quantity=product_quantity)
            session.add(new_product)
            session.commit()
            session.close()
            print(f"Product with product_name {product_name} inserted successfully.")
        except Exception as e:
                session.rollback()
                raise e
                print(f"Error occurred while inserting product: {e}")

    #Fetch all product records from the product table
    def fetch_all_products(self):
        session = Session()
        try:
            return session.query(Product).all()
        finally:
            session.close()


