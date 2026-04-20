import copy
import random
import time
from typing import List, Tuple

from Solution import Solution
from Problem import Problem
from Vehicle import Vehicle
from Carriage import Carriage

from Feasibility import IsFeasible
from Insert import InsertCustomer, BestToRoute



class BestInsert():
    def __init__(self):
        self.tryNumber = 10000

    def Solve(self, p: Problem):
        result = Solution()
        success = True
        # Construct solution
        success = self.Construct(result, p)
        if not success:
            print("Cannot generate the feasible solution!")
            return

        print("Generate the initial Solution")
        return result

    # def Construct(self, result, p):
    #     # result: Solution, p: Problem
    #     # Firstly we should get a table of all the cars with different id.
    #     # sort the vehicles by decreasing of mandatory numbers.
    #     # get all the mandatory vehicles with their id and height, and then sort
    #     vnode = copy.deepcopy(p.vehicle)
    #     vnode.sort(key=lambda n: n.var_mandatory, reverse=True)
    #
    #     # get all the carriage, and the weight corresponds with their capacity.
    #     # sort by the decreasing of capacity
    #     cnode = copy.deepcopy(result.carriage)
    #     cnode.sort(key=lambda n: n.length())
    #
    #     for t in range(self.tryNumber):
    #         complete = True
    #         for types in range(p.vehicle_types):
    #             for v_num in range(vnode[types].var_mandatory):
    #                 insert_success = False
    #                 for c in range(result.carriage_num):
    #                     c_tmp = copy.deepcopy(cnode[c])
    #                     best_insert = BestToRoute(c_tmp, vnode[types], p, True)
    #                     if best_insert[1] == -1:  # cannot accommodate the customer
    #                         continue
    #                     InsertCustomer(cnode[c], vnode[types], best_insert[1], p)
    #                     vnode[types].var_mandatory -= 1
    #                     p.vehicle[vnode[types].id].var_mandatory -= 1
    #                     vnode[types].var_optional -= 1
    #                     p.vehicle[vnode[types].id].var_optional -= 1
    #                     temp = cnode[c]
    #                     j = c
    #                     while j > 0 and cnode[j - 1].length() < temp.length():
    #                         cnode[j] = cnode[j - 1]
    #                         j -= 1
    #                     cnode[j] = temp
    #                     insert_success = True
    #                     break
    #
    #
    #     cnode.sort(key=lambda n: n.length())
    #     vnode.sort(key=lambda n: n.var_optional, reverse=True)
    #
    #     for types in range(p.vehicle_types):
    #         insert_success = False
    #         for v_num in range(vnode[types].var_optional):
    #             if vnode[types].var_optional == 0:
    #                 break
    #             for c in range(result.carriage_num):
    #                 c_tmp = copy.deepcopy(cnode[c])
    #                 best_insert = BestToRoute(c_tmp, vnode[types], p, True)
    #                 if best_insert[1] == -1:  # cannot accommodate the customer
    #                     continue
    #                 InsertCustomer(cnode[c], vnode[types], best_insert[1], p)
    #                 vnode[types].var_optional -= 1
    #                 p.vehicle[vnode[types].id].var_optional -= 1
    #                 temp = cnode[c]
    #                 j = c
    #                 while j > 0 and cnode[j - 1].length() < temp.length():
    #                     cnode[j] = cnode[j - 1]
    #                     j -= 1
    #                 cnode[j] = temp
    #                 insert_success = True
    #                 break
    #         if not insert_success:  # no feasible place for this customer
    #             break
    #     cnode.sort(key=lambda n: n.id)
    #     result.carriage = cnode
    #     result.CalculateSolutionObj(p)
    #     return True

    def Construct(self, result, p):
        # Sort the vehicles by decreasing mandatory numbers and carriages by decreasing capacity
        p.vehicle.sort(key=lambda n: n.var_mandatory, reverse=True)
        result.carriage.sort(key=lambda n: n.length(), reverse=True)

        # Iterate through vehicles and mandatory customers
        for v in p.vehicle:
            for _ in range(v.var_mandatory):
                for c in result.carriage:
                    best_insert = BestToRoute(c, v, p)
                    if best_insert[2] == -1:
                        continue
                    InsertCustomer(c, v, best_insert[2], p, floor=best_insert[1])
                    v.var_mandatory -= 1
                    v.var_optional -= 1
                    result.CalculateSolutionObj(p)
                    break

        # Sort the carriages by length and vehicles by decreasing optional numbers
        result.carriage.sort(key=lambda n: n.length())
        p.vehicle.sort(key=lambda n: n.var_optional, reverse=True)

        # Iterate through vehicles and optional customers
        for v in p.vehicle:
            for _ in range(v.var_optional):
                for c in result.carriage:
                    best_insert = BestToRoute(c, v, p)
                    if best_insert[2] == -1:
                        continue
                    InsertCustomer(c, v, best_insert[2], p, floor=best_insert[1])
                    v.var_optional -= 1
                    result.CalculateSolutionObj(p)
                    break
        return True
