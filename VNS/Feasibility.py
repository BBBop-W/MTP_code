from Conf import Config


def IsFeasible(c, p):
    if not IsFeasible_Length(c, p):
        return False
    if not IsFeasible_Height(c, p):
        return False
    return True


def IsFeasible_Length(c, p):
    positon = c.position
    if len(c.route[0]) >= 3:
        length1 = p.vehicle[c.route[0][0]].limit_length[positon]
        for i in range(1, len(c.route[0]) - 1):
            length1 += p.vehicle[c.route[0][i]].length
        length1 += p.vehicle[c.route[0][len(c.route[0]) - 1]].limit_length[positon]
        length1 += c.spacing * (len(c.route[0])-1)
        if length1 >= Config.top_length:
            return False
    if len(c.route[1]) >= 3:
        length2 = p.vehicle[c.route[1][0]].limit_length[positon]
        for i in range(1, len(c.route[1]) - 1):
            length2 += p.vehicle[c.route[1][i]].length
        length2 += p.vehicle[c.route[1][len(c.route[1]) - 1]].limit_length[positon]  # how to solve
        length2 += c.spacing * (len(c.route[1]) - 1)
        if length2 >= Config.bottom_length:
            return False

    return True


def IsFeasible_Height(c, p):
    for i in range(1, len(c.route[0]) - 1):
        if Config.top_height < p.vehicle[c.route[0][i]].height:  # ignore height constraint of the first and the last
            return False
        for i in range(1, len(c.route[1]) - 1):
            if Config.bottom_height < p.vehicle[c.route[1][i]].height:  # ignore the first and the last
                return False
    return True


def IsFeasible_route(p, route, info):
    if not IsFeasible_Length_route(p, route, info):
        return False
    if not IsFeasible_Height_route(p,route, info):
        return False
    return True


def IsFeasible_Length_route(p, route, info):
    position = info[0]
    spacing = info[1]
    floor = info[2]
    if len(route) >= 3:
        length = p.vehicle[route[0]].limit_length[position]
        for i in range(1, len(route) - 1):
            length += p.vehicle[route[i]].length
        length += p.vehicle[route[len(route) - 1]].limit_length[position]
        length += spacing * (len(route) - 1)

        if floor == 0:
            standard = Config.top_length
        else:
            standard = Config.bottom_length

        if length >= standard:
            return False

    return True


def IsFeasible_Height_route(p, route, info):
    floor = info[2]
    if floor == 0:
        standard = Config.top_height
    else:
        standard = Config.bottom_height

    for i in range(1, len(route) - 1):
        if standard < p.vehicle[route[i]].height:
            return False

    return True
