import copy
from timeit import default_timer as timer
from Carriage import Carriage
from Vehicle import Vehicle
from Conf import Config
import math
import Conf
import random
import sys
from BestInsert import BestInsert
from Solution import Solution
from Neighborhoods import Neighborhoods
from Perturb import RuinRebuid
class VNS():
    def __init__(self):
        super().__init__()
        self.max_nonImp = 0
        self.best_time = 0.0
        self.max_shake_strength = 0
        self.m_fileName = ""

    def Solve(self, result, p):
        print("Begin to solve by VNS...")
        success = False

        bi = BestInsert()

        success = bi.Construct(result, p)

        if not success:
            print("ERROR!!! -- Cannot generate feasible initial solution via BestInsert Method!")

        changed, result = self.RandomVNDBest(result, p)
        return changed, result
        # result.VerifyRoutes(p)
        # self.EvaluateSolution(result, p)

    def RandomVNDBest(self, result, p):
        print("Start BestRandomVND : ", result.obj)
        changed = False
        ns = Neighborhoods()
        method_number = 3
        method = list(range(method_number))
        random.shuffle(method)
        new_result = Solution()
        m = 0
        for iter in range(10):
            # while m <= method_number - 1:
            for m in range(3):
                flag = False
                new_result.copy_construct(result)
                if method[m] == 0:
                    flag = ns.InterRelocateBest(new_result, p)
                elif method[m] == 1:
                    flag = ns.InterSwapBest(new_result, p)
                elif method[m] == 2:
                    flag = ns.InterOptBest(new_result, p)
                if flag:
                    m = -1
                    random.shuffle(method)
                    result.copy_construct(new_result)
                    changed = True


                # m += 1










        # while True:
        #
        #     while m <= method_number - 1:
        #
        #         flag = False
        #         new_result = copy.deepcopy(result)
        #         if method[m] == 0:
        #             flag, new_result = ns.InterRelocateBest(new_result, p)
        #         elif method[m] == 1:
        #             flag, new_result = ns.InterSwapBest(new_result, p)
        #         elif method[m] == 2:
        #             flag, new_result = ns.InterOptBest(new_result, p)
        #         if flag:
        #             m = -1
        #             random.shuffle(method)
        #             result = copy.deepcopy(new_result)
        #             changed = True
        #             print("improved!!!:-)")
        #         m += 1
        #
        #     break

        print("End BestRandomVND: ", result.obj)
        return changed

    # def RandomVNDFirst(self, result, p):
    #     pass

    # def Optimization(self, result, p, num_nonImprove):
    #     print("\tStart Optimization : %d" % num_nonImprove)
    #     print("non-improve iteration = ", num_nonImprove)  # debug
    #     best = copy.deepcopy(result)
    #     now = copy.deepcopy(result)
    #     bi = BestInsert()
    #     for nonImprove in range(num_nonImprove):
    #         changed, tmp = self.RandomVNDBest(now, p)
    #
    #         # print(
    #             # f"\t------new local:{now.obj}, optimal :  {local.obj}")
    #         if tmp.obj > best.obj:
    #             best = copy.deepcopy(tmp)
    #             best.output()
    #         print("Perturb by RuinRebuid : ", nonImprove, " , best_obj = ", best.obj)  # debug
    #         strength = int(p.mandatory_sum * 0.1 + nonImprove)
    #         rebuild, tmp = RuinRebuid(result, strength, p)
    #         if rebuild is True:
    #             now = copy.deepcopy(tmp)
    #
    #             print("perturb!!! now_obj = ", now.obj)
    #             bi.Construct(now,p)
    #     result = copy.deepcopy(best)
    #     print(f"\tEnd Optimization : {result.obj}")
    #     return True, result


    # def Optimization(self, result, p, num_nonImprove):
    #     print("\tStart Optimization: %d" % num_nonImprove)
    #     print("non-improve iteration =", num_nonImprove)  # debug
    #     best = result  # Avoids unnecessary deep copy
    #     now = result  # Avoids unnecessary deep copy
    #     bi = BestInsert()  # Assumes BestInsert() initializes independently
    #
    #     for nonImprove in range(num_nonImprove):
    #         changed, tmp = self.RandomVNDBest(now, p)
    #
    #         # print(
    #         # f"\t------new local:{now.obj}, optimal :  {local.obj}")
    #         if tmp.obj > best.obj:
    #             best = tmp  # Avoids unnecessary deep copy
    #             best.output()
    #         print("Perturb by RuinRebuid:", nonImprove, ", best_obj =", best.obj)  # debug
    #         strength = int(p.mandatory_sum * 0.1 + nonImprove)
    #         rebuild, tmp = RuinRebuid(result, strength, p)
    #         if rebuild is True:
    #             now = tmp  # Avoids unnecessary deep copy
    #
    #             print("perturb!!!")
    #             bi.Construct(now, p)
    #
    #     result = best  # Avoids unnecessary deep copy
    #     print(f"\tEnd Optimization: {result.obj}")
    #     return True, result

    def Optimization(self, result, p, num_nonImprove):
        tic = timer()
        print("\tStart Optimization: %d" % num_nonImprove)
        print("non-improve iteration =", num_nonImprove)  # debug

        bi = BestInsert()  # Assumes BestInsert() initializes independently
        best = Solution()
        local_best = Solution()
        now = Solution()
        best.copy_construct(result)
        not_stop = True
        while not_stop:
            nonImprove = -1
            while nonImprove <= num_nonImprove and not_stop:
                local_best.copy_construct(result)
                nonImprove += 1
                for i in range(result.carriage_num):
                    now.copy_construct(result)
                    # self.Shake(now, nonImprove, p)
                    RuinRebuid(now, nonImprove, p)
                    self.RandomVNDBest(now, p)
                    print(f"\t------new local optimal solution : {now.obj} {local_best.obj} {result.obj}")
                    if(now.obj + Config.eps > local_best.obj):
                        local_best.copy_construct(now)
                    # Time Control

                result.copy_construct(local_best)
                if result.obj + Config.eps > best.obj:
                    best.copy_construct(result)
                    nonImprove = -1
                    print("Found solution: ", best.obj)

                toc = timer()
                if toc - tic > Config.timelimit:
                    not_stop = False
                    break

            if not_stop:
                if result.obj > best.obj:
                    result.copy_construct(best)
                print("Perturb by RuinRebuid:", nonImprove, ", best_obj =", best.obj)
                strength = int(p.mandatory_sum * 0.1 + nonImprove)
                RuinRebuid(result, strength, p)

        result.copy_construct(best)
        print(f"\tEnd Optimization: {result.obj}")
        p.Summarize()
        result.Summarize(p)
        return True, result

    # def Shake(self, result, strength, p):
    #
    #     num = min(strength/10, 4)
    #     for i in range(num)
    #         result
    #
