from typing import List, Tuple
from Problem import Problem
from Feasibility import IsFeasible
from Vehicle import Vehicle
import copy


# def InsertCustomer(c, v, place, p, BothFloor = True, floor = 0):
#     c_temp = copy.deepcopy(c)
#     c_temp.route[floor].insert(place, v.id)
#
#     if IsFeasible(c_temp, p):
#         c.route[floor].insert(place, v.id)
#         c.CalculateCarriageObj(p)
#         return True
#     elif BothFloor == True:
#         c_temp = copy.deepcopy(c)
#         c_temp.route[1-floor].insert(place, v.id)
#         if IsFeasible(c_temp, p):
#             c.route[1-floor].insert(place, v.id)
#             return True
#     c.CalculateCarriageObj(p)
#     return False


def InsertCustomer(c, v, place, p, BothFloor=True, floor=0):
    c.route[floor].insert(place, v.id)
    if IsFeasible(c, p):
        c.CalculateCarriageObj(p)
        return True
    else:
        c.route[floor].pop(place)
        if BothFloor:
            c.route[1 - floor].insert(place, v.id)
            if IsFeasible(c, p):
                c.CalculateCarriageObj(p)
                return True
            else:
                c.route[1 - floor].pop(place)
        c.CalculateCarriageObj(p)
        return False


# def InsertBackCustomer(r, cid, p, feasible):
#     place = len(r) - 1
#     InsertCustomer(r, cid, place, p)
#     place = len(r) - 1
#     return InsertCustomer(r, cid, place, p, feasible)


def BestToRoute(c, v, p):
    max_obj = 0
    best_place = -1
    best_floor = -1
    if c.length(0) == 0:
        return max_obj, 0, 0
    if c.length(1) == 0:
        return max_obj, 1, 0
    for f in [0, 1]:
        for i in range(len(c.route[f])):
            obj_temp = c.obj
            if InsertCustomer(c, v, i, p, floor=f, BothFloor=False):
                obj = c.obj - obj_temp
                if obj > max_obj:
                    max_obj = obj
                    best_place = i
                    best_floor = f
                EraseVehicle(c, i, p, f)
    return max_obj, best_floor, best_place


def EraseVehicle(c, place, p, floor=0):
    v_id = c.route[floor][place]
    if c.length(floor) == 0:
        return False
    c.route[floor].pop(place)
    if IsFeasible(c, p):
        c.CalculateCarriageObj(p)
        return True
    else:
        c.route[floor].insert(place, v_id)
        return False
