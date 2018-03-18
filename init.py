import math
import random

class City(object):
   def __init__(self, city_id, x = None, y = None, capacity = None):
      self.city_id = city_id
      self.x = random.randint(1, 200) if not x else x
      self.y = random.randint(1, 200) if not y else y
      self.capacity = random.randint(1, 200) if not capacity else capacity
      
   def same_city(self, another):
      return another.city_id == self.city_id
   
class TourManager(object):
   def __init__(self, num_cars, capacity):
      self.destinationCities = []
      self.num_cars = num_cars
      self.car_limit = capacity
      
def init(fileName):
   with open(fileName) as file:
      content = file.readlines() 
      content = filter(None, [x.strip() for x in content]) 
      cities = int(content[0])
      cars = int(content[1])
      car_limit = int(content[2])
      tourManager = TourManager(cars, car_limit)

      for i in range(0, cities):
         tmp = content[3 + i].split()
         x = int(tmp[1])
         y = int(tmp[2])
         tmp = content[3 + cities + i].split()
         capacity = int(tmp[1])
         tourManager.destinationCities.append(City(i, x, y, capacity)) 
      return tourManager
