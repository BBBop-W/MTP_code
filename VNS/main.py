import copy

from Problem import Problem
from BestInsert import BestInsert
from VNS import VNS

p = Problem()
p.LoadVRPTW('1')

b = BestInsert()
result = b.Solve(p)

V = VNS()
changed, result2 = V.Optimization(result,p,20)



