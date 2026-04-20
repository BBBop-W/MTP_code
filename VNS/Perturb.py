import random
from typing import List
from Problem import Problem
from Solution import Solution
from Neighborhoods import Neighborhoods
from Conf import Config
import copy
from BestInsert import BestInsert

# class Perturb:
#     @staticmethod
#     def BlockExchange(result, strength, p):
#         nb = Neighborhoods()
#         old = Solution(result)
#         for i in range(strength):
#             while not nb.BlockExchangeRandom(result, 4, p):
#                 result = Solution(old)
#         return True
#
#     @staticmethod
#     def RuinRebuid(result: Solution, strength: int, p: Problem) -> bool:
#         old = Solution(result)
#         strength = min(strength, p.GetCustomerNumber() // 2)
#         unserve = []
#         for i in range(strength):
#             r = -1
#             while result[r].Empty():
#                 r = random.randint(0, len(result) - 1)
#             place = random.randint(1, len(result[r]) - 2)
#             unserve.append(result[r][place])
#             Solution.EraseCustomer(result[r], place, p)
#         bi = BestInsert()
#         if bi.Complete(result, p, unserve):
#             print("======Reconstruct Solution Success!!!  ")
#             return True
#         result = Solution(old)
#         return False
#


def RuinRebuid(result, strength, p):
    old = Solution()
    old.copy_construct(result)
    strength = min(strength, result.carriage_num // 2 + 1)
    Ns = Neighborhoods()
    for i in range(strength):
        r = random.randint(0, result.carriage_num - 1)
        is_success= Ns.Reposition(result, r ,p)

        if is_success is False:
            print("shake false!!!")
            return False

    bi = BestInsert()
    if bi.Construct(result, p):
        print("======Reconstruct Solution Success!!!  ")
        return True
    result.copy_construct(old)
    return False

