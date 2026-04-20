class Vehicle:
    def __init__(self):

        self.id = 0
        self.brand = ''
        self.model = ''
        self.length = 0.0
        self.height = 0.0

        self.num_optional = 0
        self.num_mandatory = 0

        self.var_mandatory = 0
        self.var_optional = 0

        self.limit_length = [0,0]

        # self.position_carriage = 0  # which carriage
        #TODO

    def UpdateParameter_Removing(self):

        if self.var_mandatory > 0:
            self.var_mandatory += 1
        elif self.var_optional == self.num_optional - self.num_mandatory:
            self.var_mandatory += 1
        else:
            self.var_optional += 1



