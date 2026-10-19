#Product Management System Using PostgreSQL CRUD Operations
from prdDBOperations import Product

class ProductInfo:
    def __init__(self, product_id, product_name, product_price, product_quantity):
        self.product_id = product_id
        self.product_name = product_name
        self.product_price = product_price
        self.product_quantity = product_quantity

#Create an instance of Product class to perform database operations
product_db = Product()

#Create the product table in the database
product_db.create_table()

#Insert product records into the product table
product_name=input("Enter the product name: ")
product_price=float(input("Enter the product price: "))
product_quantity=int(input("Enter the product quantity: "))
product_db.insert_product(product_name, product_price, product_quantity)    

