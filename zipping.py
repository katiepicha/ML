list1 = ['a','b','c']
list2 = [1,2,3]
list3 = [1.5, 3.1, 5.7]

# zip function allows you to take multiple iterable objects and iterate through them at the same time
for item in zip(list1, list2, list3):
    l1,l2,l3 = item
    print(l1)
    print(l2)
    print(l3)