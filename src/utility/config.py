

class config:

    eps = 0.0001
    gap = 0.00005  # 0.005%
    timelimit = 900
    max_units_per_compartment = 10
    max_wagon_pricing_columns = 20
    max_compartment_pricing_columns_per_subproblem = 20
    gurobi_threads = 1
    safety_clearance_delta = 400.0

    @classmethod
    def apply_gurobi_params(cls, model):
        if cls.gurobi_threads is not None:
            model.Params.Threads = int(cls.gurobi_threads)

    A_len = 4300.0
    B_len = 2000.0
    C_len = 12400.0
    D_len = 5000.0
    E_len = 14900.0

    A_height_h = 1700.0
    A_height_m = 1780.0

    B_height = 2100.0
    C_height = 2270.0

    E_height = 2070.0
    D_height_h = 2070.0
    D_height_m = 1720.0

    bottom_len = 2*A_len + 2*B_len + C_len
    top_len = 2*D_len + E_len




