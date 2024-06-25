inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output=[]

#@ Checking for conditions in ReLu:

for i in inputs:
    if i>0:
        output.append(i)
    else:
        output.append(0)


print(output)
