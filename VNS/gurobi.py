import json

import gurobipy as gp
import pandas as pd
from Conf import Config

car_info = pd.read_csv("cars.csv", index_col=['Brand', 'Model'])

i_size = car_info.index.to_list()
j_size = Config.carriage_num

theta, phi, alpha, beta, gamma, epsilon = {}, {}, {}, {}, {}, {}

I = []
idx = 1
for i in i_size:

    theta[idx] = 1 if car_info.loc[i, 'Height'] < Config.top_height else 0  # 𝑖类商品车高度是否小于上层厢车高度
    phi[idx] = 1 if car_info.loc[i, 'Height'] < Config.slope3_height else 0  # i类商品车高度是否小于上层层厢车slope3处高度
    alpha[idx] = 1 if car_info.loc[i, 'Height'] < Config.bottom_height else 0  # 𝑖类商品车高度是否小于下层厢车凹槽处高度
    beta[idx] = 1 if car_info.loc[i, 'Height'] < Config.slope1_height else 0  # i类商品车高度是否小于下层slope1高度
    gamma[idx] = 1 if car_info.loc[i, 'Height'] < Config.slope2_height_1 else 0  # i类商品车高度是否小于下层slope2高度
    epsilon[idx] = 1 if car_info.loc[i, 'Height'] < Config.slope2_height_2 else 0  # i类商品车高度是否小于下层slope2高度
    I.append(idx)
    idx += 1


J = [j for j in range(1, j_size + 1)]
K = [1, 2]

L1 = 12400.0
L2 = 2000.0
L3 = 4300.0
L4 = 5000.0
L5 = 14900.0
Delta = 400
BigM = gp.GRB.MAXINT


try:
    m = gp.Model('railway_problem')
    Pi = m.addVars(J, vtype=gp.GRB.BINARY, name='Pi')

    U = m.addVars(I, J, vtype=gp.GRB.INTEGER, lb=0, ub=10, name='U')
    W = m.addVars(I, J, K, vtype=gp.GRB.INTEGER, lb=0, ub=10, name='W')
    X = m.addVars(I, J, vtype=gp.GRB.INTEGER, lb=0, ub=10, name='X')
    Y = m.addVars(I, J, K, vtype=gp.GRB.INTEGER, lb=0, ub=10, name='Y')
    Z = m.addVars(I, J, K, vtype=gp.GRB.INTEGER, lb=0, ub=10, name='Z')
    delta = m.addVars(I, vtype=gp.GRB.INTEGER, lb=0, ub=10, name='delta')
    m.update()

    # objective
    m.setObjective(sum(delta[i] for i in I), sense=gp.GRB.MAXIMIZE)

    # constraint (2)
    for i in I:
        m.addConstr(delta[i] <= (car_info['Optional#'].iloc[i-1] - car_info['Mandatory#'].iloc[i-1]))

    # constraint (3)
    for i in I:
        m.addConstr(sum(X[i, j] + sum(Y[i, j, k] + Z[i, j, k] + W[i, j, k] for k in K) for j in J) >=
                    car_info['Mandatory#'].iloc[i-1] + delta[i])

    # constraint (4)
    for j in J:
        m.addConstr(
            sum((X[i, j] + sum(Y[i, j, k] + Z[i, j, k] for k in K)) * (Delta + car_info['Length'].iloc[i-1]) for i in I) <=
            L1 + 2 * L2 + 2 * L3 - Delta)

    # constraint (5)
    for j in J:
        for k in K:
            m.addConstr(
                sum((X[i, j] + Y[i, j, k] + Z[i, j, k]) * (Delta + car_info['Length'].iloc[i-1]) for i in I) <=
                L1 + L2 + L3)

    # constraint (6)
    for j in J:
        for k in K:
            m.addConstr(
                sum((X[i, j] + Y[i, j, k]) * (Delta + car_info['Length'].iloc[i-1]) for i in I) <= L1 + L2 + Delta)

    # constraint (7)
    for j in J:
        m.addConstr(
            sum(X[i, j] * (Delta + car_info['Length'].iloc[i-1]) for i in I) <= L1+ Delta)

    # constraint (8)
    for j in J:
        m.addConstr(
            sum((U[i, j] + sum(W[i, j, k] for k in K))* (Delta + car_info['Length'].iloc[i-1]) for i in I) <=
            2 * L4 + L5 - Delta)

    # constraint (9)
    for j in J:
        for k in K:
            m.addConstr(
                sum((U[i, j] + W[i, j, k]) * (Delta + car_info['Length'].iloc[i-1]) for i in I) <=
                L4 + L5 + BigM * (1 - Pi[j]))

    # constraint (10)
    for j in J:
        m.addConstr(
            sum((U[i, j]) * (Delta + car_info['Length'].iloc[i-1]) for i in I) <=
            L5 + Delta + BigM * (1 - Pi[j]))

    # constraint (11)
    m.addConstrs(X[i, j] <= BigM * alpha[i] for i in I for j in J)

    # constraint (12)
    m.addConstrs(Y[i, j, k] <= BigM * beta[i] for i in I for j in J for k in K)

    # constraint (13)
    m.addConstrs(Z[i, j, k] <= BigM * gamma[i] + BigM * Pi[j] for i in I for j in J for k in K)

    # constraint (14)
    m.addConstrs(Z[i, j, k] <= BigM * epsilon[i] + BigM * (1 - Pi[j]) for i in I for j in J for k in K)

    # constraint (15)
    m.addConstrs(U[i, j] <= BigM * theta[i] for i in I for j in J)

    # constraint (16)
    m.addConstrs(W[i, j, k] <= BigM * theta[i] + BigM * Pi[j] for i in I for j in J for k in K)

    # constraint (17)
    m.addConstrs(W[i, j, k] <= BigM * phi[i] + BigM * (1 - Pi[j]) for i in I for j in J for k in K)

    m.Params.LogToConsole = True
    m.optimize()

    if m.status == gp.GRB.Status.INFEASIBLE:
        print('Optimization was stopped with status %d' % m.status)
        m.computeIIS()
        m.write("model0223.ilp")

    if m.status == gp.GRB.Status.INFEASIBLE:
        print('Optimization was stopped with status %d' % m.status)
        # do IIS, find infeasible constraints
        m.computeIIS()
        print('\nThe following constraint cannot be satisfied:')
        removed = []
        for c in m.getConstrs():
            if c.IISConstr:
                print('%s' % c.ConstrName)
            removed.append(str(c.ConstrName))
        m.write("model0223.ilp")
    print("Optimal objective value is %g" % m.objVal)

    if m.status == gp.GRB.Status.OPTIMAL:
        print('optimal!')



except gp.GurobiError as exception:
    print('Error code ' + str(exception.errno) + ": " + str(exception))


except AttributeError:
    print('Encountered an attribute error')



sol = m.getAttr('X', delta)
car_sol = pd.DataFrame({"brand": [i[0] for i in i_size], "type": [i[1] for i in i_size], "num_chosen": sol.values()})
car_sol.to_csv("output/car_sol.csv", index=False)

sol_U = m.getAttr('X', U)
sol_W = m.getAttr('X', W)
sol_X = m.getAttr('X', X)
sol_Y = m.getAttr('X', Y)
sol_Z = m.getAttr('X', Z)
sol_Pi = m.getAttr('X', Pi)
carriage_list = []

for j in J:
    car_dict_top,car_dict_slope3_left, car_dict_slope3_right, car_dict_slope1_left, car_dict_slope1_right, car_dict_slope2_left, car_dict_slope2_right, car_dict_bottom ={},{}, {}, {}, {}, {}, {}, {}
    for i in I:
        key = i_size[i-1]
        if sol_U[i, j] > 0:
            car_dict_top['-'.join(key)] = sol_U[i, j]
        if sol_W[i, j, 1] > 0:
            car_dict_slope3_left['-'.join(key)] = sol_W[i, j, 1]
        if sol_W[i, j, 2] > 0:
            car_dict_slope3_right['-'.join(key)] = sol_W[i, j, 2]
        if sol_X[i, j] > 0:
            car_dict_bottom['-'.join(key)] = sol_X[i, j]
        if sol_Y[i, j, 1] > 0:
            car_dict_slope1_left['-'.join(key)] = sol_Y[i, j, 1]
        if sol_Y[i, j, 2] > 0:
            car_dict_slope1_right['-'.join(key)] = sol_Y[i, j, 2]
        if sol_Z[i, j, 1] > 0:
            car_dict_slope2_left['-'.join(key)] = sol_Z[i, j, 1]
        if sol_Z[i, j, 2] > 0:
            car_dict_slope2_right['-'.join(key)] = sol_Z[i, j, 2]
    if sol_Pi[j] > 0.5:
        position = 'middle'
    else:
        position = 'horizontal'
    carriage_list.append(
        {'top': car_dict_top, 'position' : position,
         'slope3_left': car_dict_slope3_left, 'slope3_right': car_dict_slope3_right,
         'slope2_left': car_dict_slope2_left, 'slope2_right': car_dict_slope2_right,
         'slope1_left': car_dict_slope1_left, 'slope1_right': car_dict_slope1_right,
         'bottom': car_dict_bottom})

with open("output/carriage_info.json", 'w', encoding='utf-8') as f:
    json.dump({'carriage': carriage_list}, f, indent=4, ensure_ascii=False)
