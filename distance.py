import random
unit_distance = {}
list1 = range(40)
#print list1
for i in range(40):
    for j in range(40):
        if list1[i] == list1[j]:
            pass
        else:
            unit_distance[list1[i],list1[j]] = \
            unit_distance[list1[j],list1[i]] = random.randint(50, 500)
print unit_distance
