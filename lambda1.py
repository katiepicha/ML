import os

# clears the screen
clear = lambda: os.system("cls")
clear()


# normal function
def remainder(num):
    return num % 2
print(remainder(5))

# lambda function
remainder = lambda num: num % 2
print(remainder(5))
print(type(remainder)) # returns function because lambda creates a function called remainder


# multiple arguments in lambda function
product = lambda x,y: x * y
print(product(2,3))


# lambda functions within a function
def myfunction(num):
    return lambda x: x * num

result10 = myfunction(10) # result 10 is a function created from a lambda that requires 1 argument
# result10 = lambda x: x * 10 -- performs the same as the line above
result100 = myfunction(100)

print(result10(9))
print(result100(9))


def myfunc(n):
    return lambda a: a * n

mydoubler = myfunc(2) # 2 is n
mytripler = myfunc(3) # 3 is n
# when we call these function, the argument we give it is a
print(mydoubler(11)) # 11 is a
print(mytripler(11)) # 11 is a


numbers = [2,4,6,8,10,3,18,14,21]

# filter function
filtered_list = list(filter(lambda num: (num > 7), numbers)) # first argument is lambda function, second argument is the iterable (list)
print(filtered_list)

# map function - applies lambda function to each element in the list
mapped_list = list(map(lambda num: num % 2, numbers))
print(mapped_list)


# other examples
x = lambda a: a + 10
print(x(5))

x = lambda a, b, c: a + b + c
print(x(5, 6, 7))


# traditional way
def addition(n):
    return n + n

numbers = [1,2,3,4]

result = map(addition, numbers)
print(list(result))

# without external function
result = map(lambda num: num + num, numbers)
print(list(result))