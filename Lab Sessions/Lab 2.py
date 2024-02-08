##excerise 1 - practice funtions
# find max of a number in a list
import numpy as np
import matplotlib.pyplot as plt

my_list = [4, 1, 1, 7, 55, 2, 2, 10, 24]

def find_max(list):
    for i in range(len(list)):
        max_number = list[i]

        for j in range(len(list) - 1):
            if max_number > list[j + 1]:
                max_check = True

            else:
                max_check = False

        if max_check is True:
            print("max number = %d" % max_number)

def sum_list(list):
    sum = 0

    for i in range(len(list)):
        sum = sum + list[i]
    
    return sum

def reverse_list(list):
    reversed_list = []

    for i in range(len(list) - 1):
        reversed_list = [list[i]] + reversed_list
    
    return reversed_list

def shuffle_list(list):
    shuffled_list = np.zeros(len(list))
    #print(shuffled_list)
    order = np.random.permutation(len(list))[:len(list) - 1]
    #print(order)

    for i in range(len(list) - 1):
        new_position = order[i]
        shuffled_list[new_position] = list[i]

    return shuffled_list

def unique_elements(list):
    ##TODO: fix this thing
    current_list = list
    unique_list = []
    
    for i in range(len(current_list) - 1):
        target_number = current_list[i]

        for j in range(len(current_list) - 1):
            if i == j:
                pass

            else:
                print(current_list)
                if target_number == current_list[j]:
                    current_list.pop(j)

                    current_list = current_list

    return current_list


##excersise 4:
def summation(x, n):
    sum = 0
    the_range = np.arange(1, n + 1)

    for i in the_range:
        nth_term = (1/(2 * i - 1) ** 2) * np.cos((2 * i - 1) * x)

        sum = sum + nth_term

    return sum

def residuals(x, j):
    R = summation(x, j + 1) - summation(x, j)

    return R

################################
print('the list =', my_list)
print('max value =', find_max(my_list))
print('the sum =', sum_list(my_list))
print('the reverse =', reverse_list(my_list))
print('shuffled list =', shuffle_list(my_list))
print('unique list =', unique_elements(my_list))

fig, ax = plt.subplots(1,2)
for k in range(2, 22, 2):
    x = np.linspace(0, 4 * np.pi, 100)

    ax[0].plot(x, summation(x, k), label = 'n = %d' % k)
    ax[1].plot(x, residuals(x, k), label = 'j = %d' % k)

ax[0].set_title('plot of f(x, n)')
ax[1].set_title('plot of residuals R(x, j)')
ax[0].legend(loc = 'upper right')
ax[1].legend(loc = 'upper right')
fig.suptitle('excersise 4 - graphically plotting a summation')


plt.show()

##numerically:
for k in range(1, 20):
    x = np.arange(0, 4 * np.pi, np.pi)

    for i in range(len(x)):
        print('x =', x[i], 'n, j =', k)
        print('summation = ',summation(x[i], k))
        print('residual = ', residuals(x[i], k))
        print()




    
