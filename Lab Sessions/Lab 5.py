import numpy as np
import random
import string
import matplotlib.pyplot as plt

hundred_random_numbers = [0] * 100

for k in range(len(hundred_random_numbers)):
    hundred_random_numbers[k] = random.random()

print(hundred_random_numbers)

three_integers = [0] * 3

def divisible_by_three(number):
    return number % 3 == 0
 
for i in range(len(three_integers)):
    number = random.randint(1000, 3500)
        
    while not divisible_by_three(number):
        number = random.randint(1000, 3500)
        
    three_integers[i] = number
            
print(three_integers)

def lottery_winner(tickets):
    lottery_tickets = [0] * tickets 
    for i in range(0, tickets):
        lottery_tickets[i] = random.randint(100000, 999999)
    
    winner = lottery_tickets[random.randint(0, tickets)]

    print('winner is number:', winner) 

lottery_winner(200)

def random_word(length):
   letters = string.ascii_lowercase
   
   return ''.join(random.choice(letters) for i in range(length))

hundred_char = random_word(100)

print(random.choice(hundred_char) for i in range(len(hundred_char)))

def random_3d_coords(coord_range):
    start, stop = coord_range
    coords = [0] * 3
    
    for i in range(0, 3):
        coords[i] = random.uniform(start, stop)
        
    return coords
        
def coordinate_check(coords):
    ux, uy, uz = coords
    
    if np.sqrt((ux)**2 + (uy)**2 + (uz)**2) > 1:
        return False
    
    else:
        return True

collection_coords = []

for i in range(0, 1000):
    coord = random_3d_coords((-1, 1))
        
    while coordinate_check(coord) is False:
        coord = random_3d_coords((-1, 1))
        
    collection_coords.append(coord)

collection_coords = np.array(collection_coords)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(collection_coords[:, 0], collection_coords[:, 1], collection_coords[:, 2], marker='o')
plt.show()

for i in range(len(collection_coords[:])):
    collection_coords