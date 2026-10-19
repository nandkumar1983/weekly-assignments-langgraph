class Vehicle:
    def show_details(self):
        print("This is a vehicle.")


class Car(Vehicle):
    def show_details(self):
        print("This is a Car with 4 wheels.")


class Bike(Vehicle):
    def show_details(self):
        print("This is a Bike with 2 wheels.")


car = Car()
bike = Bike()

car.show_details()
bike.show_details()
