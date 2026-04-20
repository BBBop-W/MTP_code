import copy
import random
from Solution import Solution
from Conf import Config
from Feasibility import IsFeasible, IsFeasible_route
from Insert import InsertCustomer, BestToRoute, EraseVehicle

class Neighborhoods():
    def __init__(self):
        self.try_num = 100

    def RelocateBasic(self, result, c1_id, old_place, c2_id, new_place, old_floor = 0, new_floor = 0, p=None):
        if c1_id == c2_id and (old_place == new_place or old_place == new_place + 1 or old_place + 1 == new_place) and old_floor==new_floor:
            return False
        v_id = result.carriage[c1_id].route[old_floor][old_place]
        vehicle = p.GetVehicle(v_id)
        if not EraseVehicle(result.carriage[c1_id], old_place, p, old_floor):
            return False
        if not InsertCustomer(result.carriage[c2_id], vehicle, new_place, p, BothFloor=False, floor = new_floor):
            InsertCustomer(result.carriage[c1_id], vehicle, old_place, p, BothFloor=False, floor=old_floor)
            return False

        result.CalculateSolutionObj(p)
        return True

    def SwapBasic(self, result, c1_id, c2_id, place1, place2, p, floor1=0, floor2=0):
        if c1_id == c2_id and (place1 == place2 or place1 + 1 == place2 or place1 == place2 + 1) and floor1==floor2:
            return False
        result_tmp = Solution()
        result_tmp.copy_construct(result)
        tmp = result_tmp.carriage[c1_id].route[floor1][place1]
        result_tmp.carriage[c1_id].route[floor1][place1] = result_tmp.carriage[c2_id].route[floor2][place2]
        result_tmp.carriage[c2_id].route[floor2][place2] = tmp

        is_success = IsFeasible(result_tmp.carriage[c1_id], p) and IsFeasible(result_tmp.carriage[c2_id], p)
        if is_success:
            tmp = result.carriage[c1_id].route[floor1][place1]
            result.carriage[c1_id].route[floor1][place1] = result.carriage[c2_id].route[floor2][place2]
            result.carriage[c2_id].route[floor2][place2] = tmp
            result.CalculateSolutionObj(p)
            return True
        else:
            return False

    def Reposition(self, result, c_id, p):

        result_tmp = Solution()
        result_tmp.copy_construct(result)

        if result_tmp.carriage[c_id].position == 0: # horizontal
            length = len(result_tmp.carriage[c_id].route[0])
            if length >= 4:
                v1_id = result_tmp.carriage[c_id].route[0][0]
                v2_id = result_tmp.carriage[c_id].route[0][length - 1]
                EraseVehicle(result_tmp.carriage[c_id], length - 1, p, 0)
                EraseVehicle(result_tmp.carriage[c_id], 0, p, 0)
                vehicle1 = p.GetVehicle(v1_id)
                vehicle1.UpdateParameter_Removing()
                vehicle2 = p.GetVehicle(v2_id)
                vehicle2.UpdateParameter_Removing()
            result_tmp.carriage[c_id].position = 1

        elif result_tmp.carriage[c_id].position == 1: # middle
            length = len(result_tmp.carriage[c_id].route[1])
            if length >= 4:
                v1_id = result_tmp.carriage[c_id].route[1][0]
                v2_id = result_tmp.carriage[c_id].route[1][length - 1]
                EraseVehicle(result_tmp.carriage[c_id], length - 1, p, 1)
                EraseVehicle(result_tmp.carriage[c_id], 0, p, 1)
                vehicle1 = p.GetVehicle(v1_id)
                vehicle1.UpdateParameter_Removing()
                vehicle2 = p.GetVehicle(v2_id)
                vehicle2.UpdateParameter_Removing()
            result_tmp.carriage[c_id].position = 0
        result_tmp.CalculateSolutionObj(p)
        is_success = IsFeasible(result_tmp.carriage[c_id], p)

        if is_success:
            result.copy_construct(result_tmp)
        return is_success

    def CrossBasic(self, result, p, c1_id, c2_id, place1, place2,floor1=0,floor2=0):
        len1 = len(result.carriage[c1_id].route[floor1])
        len2 = len(result.carriage[c2_id].route[floor2])

        if len1 <= 2 or len2 <= 2:
            return False
        if place1 > len1-1 or place2 > len2-1:
            return False
        if c1_id == c2_id and floor1 == floor2:
            return False

        route1 = result.carriage[c1_id].route[floor1]
        position1 = result.carriage[c1_id].position
        spacing1 = result.carriage[c1_id].spacing
        info1 = (position1, spacing1, floor1)

        route2 = result.carriage[c2_id].route[floor2]
        position2 = result.carriage[c2_id].position
        spacing2 = result.carriage[c2_id].spacing
        info2 = (position2, spacing2, floor2)

        tmp = route1[place1:]
        route1 = route1[:place1] + route2[place2:]
        route2 = route2[:place2] + tmp

        is_success1 = IsFeasible_route(p, route1, info1)
        is_success2 = IsFeasible_route(p, route2, info2)

        if is_success1 and is_success2:
            result.carriage[c1_id].route[floor1] = route1
            result.carriage[c2_id].route[floor2] = route2
            result.CalculateSolutionObj(p)
            return True
        else:

            return False

    # def IntraRelocateBest(self, result, index, p):
    #     best, old = result.copy(), result.copy()
    #     improved = False
    #     for j in range(1, len(result[index]) - 1):
    #         for n in range(len(result[index]) - 1):
    #             if self.RelocateBasic(result, index, j, index, n,
    #                              p) and result.m_penalty_obj + Config.eps < best.m_penalty_obj:
    #                 best, improved = result.copy(), True
    #             result = old.copy()
    #
    #     result = best.copy()
    #     return improved
    #
    # def IntraSwapBest(self, result, index, p):
    #     best, old = result.copy(), result.copy()
    #     improved = False
    #     for j in range(1, len(result[index]) - 1):
    #         for n in range(j + 1, len(result[index]) - 1):
    #             if self.SwapBasic(result, index, j, index, n, p) and result.m_penalty_obj + Config.eps < best.m_penalty_obj:
    #                 best, improved = result.copy(), True
    #             result = old.copy()
    #
    #     result = best.copy()
    #     return improved
    #
    # def IntraOptBest(self, result, p):
    #     best, old = result.copy(), result.copy()
    #     improved = False
    #     for i in range(len(result)):
    #         now = result.copy()
    #         if self.IntraOptBest(now, i, p):
    #             if now.m_penalty_obj + Config.eps < best.m_penalty_obj:
    #                 best = now
    #                 improved = True
    #     result = best
    #     return improved
    #
    def InterRelocateBest(self, result, p):
        best = Solution()
        old = Solution()
        best.copy_construct(result)
        old.copy_construct(result)
        improved = False
        length = result.carriage_num
        for i in range(length):
            for j in range(length):
                for f1 in [0,1]:
                    for f2 in [0,1]:
                        for m in range(len(result.carriage[i].route[f1])):
                            for n in range(len(result.carriage[j].route[f2])):
                                if self.RelocateBasic(result, i,m, j, n,f1,f2, p) and result.obj  > best.obj:
                                    best.copy_construct(result)
                                    improved = True
                                result.copy_construct(old)

        result.copy_construct(best)
        return improved

    def InterSwapBest(self, result, p):
        best = Solution()
        old = Solution()
        best.copy_construct(result)
        old.copy_construct(result)
        improved = False
        length = result.carriage_num
        for i in range(length):
            for j in range(length):
                for f1 in [0,1]:
                    for f2 in [0,1]:
                        for m in range(len(result.carriage[i].route[f1])):
                            for n in range(len(result.carriage[j].route[f2])):
                                if self.SwapBasic(result, i,j, m, n,p,f1,f2) and result.obj  > best.obj:
                                    best.copy_construct(result)
                                    improved = True
                                result.copy_construct(old)

        result.copy_construct(best)
        return improved

    def InterOptBest(self, result, p):
        best = Solution()
        old = Solution()
        best.copy_construct(result)
        old.copy_construct(result)
        improved = False
        length = result.carriage_num
        for i in range(length):
            for j in range(length):
                for f1 in [0,1]:
                    for f2 in [0,1]:
                        for m in range(len(result.carriage[i].route[f1])):
                            for n in range(len(result.carriage[j].route[f2])):
                                if self.CrossBasic(result,p, i,j,m,n,f1,f2) and result.obj  > best.obj:
                                    best.copy_construct(result)
                                    improved = True
                                result.copy_construct(old)

        result.copy_construct(best)

        return improved
