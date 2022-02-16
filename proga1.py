import numpy

def f(x):
    y = numpy.cos(x)
    return y

for i in range(5):
    res = f(i)
    print(res)
