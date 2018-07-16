from hv import HyperVolume
referencePoint = [2, 2]
hyperVolume = HyperVolume(referencePoint)
front = [[1, 0], [0, 1]]
result = hyperVolume.compute(front)
print result
