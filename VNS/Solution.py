# Solution.py
import json
from typing import List
from Problem import Problem
from typing import Optional
import sys
from Conf import Config
import math

class Carriage():
    def __init__(self):
        self.id = 0
        self.spacing = 400
        self.route = [0,0]
        self.route[0] = []  # up
        self.route[1] = []  # down
        self.obj = 0.0  # maximization
        self.num = 0
        self.position = 0  # horizontal

    def length(self, floor=-1):
        if floor == -1:
            return len(self.route[0]) + len(self.route[1])
        else:
            return len(self.route[floor])
        # return len(self.route)

    def CalculateCarriageObj(self, p):
        SpaceSum = 0
        LengthSum = 0
        Num1 = len(self.route[0])
        # for i in range(Num):
        #     v = p.GetVehicle(self.route[0][i])
        Num2 = len(self.route[1])
        self.obj = (Num1 + Num2)*(Num1+Num2)
        length1 = 0
        for i in range(len(self.route[0])):
            length1 += p.vehicle[self.route[0][i]].length
        length2 = 0
        for i in range(1, len(self.route[1])):
            length2 += p.vehicle[self.route[1][i]].length
        # self.obj = (length1 + length2)*(length1 + length2)//1000000
        spacing_sum = (Config.top_length - length1)**2 + (Config.bottom_length - length2)**2
        self.obj = -spacing_sum//100000
        return self.obj  # 暂定

    def copy_construct(self, origin):
        self.id = origin.id
        self.spacing = origin.spacing
        self.route[0] = [i for i in origin.route[0]]
        self.route[1] = [i for i in origin.route[1]]
        self.obj = origin.obj
        self.num = origin.num
        self.position = origin.position  # horizontal


class Solution():
    def __init__(self):
        self.carriage_num = Config.carriage_num
        self.carriage = [Carriage() for _ in range(self.carriage_num)]
        for i in range(self.carriage_num):
            self.carriage[i].id = i
        self.obj = 0

    def CalculateSolutionObj(self, p):
        objective = 0
        for i in range(self.carriage_num):
            self.carriage[i].CalculateCarriageObj(p)
            objective += self.carriage[i].obj
        self.obj = objective
        return self.obj

    def output(self):
        for i in range(self.carriage_num):
            print("carriage", i)
            print("up:", self.carriage[i].route[0])
            print("down:", self.carriage[i].route[1])
            print("obj:", -self.carriage[i].obj)
            print()

    def copy_construct(self, origin):
        self.obj = origin.obj
        self.carriage_num = origin.carriage_num
        for i in range(self.carriage_num):
            self.carriage[i].copy_construct(origin.carriage[i])

    def Summarize(self, p):
        carriage_list = []

        for C in self.carriage:
            position = 'horizontal' if C.position == 0 else 'middle'
            top = []
            for idx in C.route[0]:
                top.append(' '.join([p.GetVehicle(idx).brand, p.GetVehicle(idx).model]))
            down = []
            for idx in C.route[1]:
                down.append(' '.join([p.GetVehicle(idx).brand, p.GetVehicle(idx).model]))

            carriage_list.append(
                {'position': position,
                 'top': top,
                 'bottom': down})

        with open("outputVNS/carriage_info.json", 'w', encoding='utf-8') as f:
            json.dump({'carriage': carriage_list}, f, indent=4, ensure_ascii=False)
