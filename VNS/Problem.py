from Vehicle import Vehicle
import pandas as pd
class Problem:

    def __init__(self):
        self.vehicle_types = 0
        self.vehicle = [] # list of object Vehicle
        self.mandatory_sum = 0
        self.optional_sum = 0

    def GetVehicle(self, id):
        for i in range(self.vehicle_types):
            if self.vehicle[i].id == id:
                return self.vehicle[i]

    def LoadVRPTW(self, file_name):
        inputfile = "cars.csv"
        fin = open(inputfile, 'r')
        line = fin.readline()
        i = 0
        while True:
        # for i in range(self.vehicle_types):

            line = fin.readline()
            if not line:
                break
            line = line.split(',')
            V = Vehicle()
            V.id = i
            V.brand = line[0]
            V.model = line[1]
            V.length = float(line[2])
            V.height = float(line[3])
            V.num_optional = int(line[4])
            V.num_mandatory = int(line[5])
            V.limit_length[0] = int(line[6])
            V.limit_length[1] = int(line[7])
            V.var_optional = int(line[4])
            V.var_mandatory = int(line[5])
            self.vehicle.append(V)
            i += 1
            self.mandatory_sum += V.num_mandatory
            self.optional_sum += V.num_optional

        self.vehicle_types = i
        fin.close()

    def Summarize(self):
        car_sol = pd.DataFrame(
            {"brand": [v.brand for v in self.vehicle], "model": [v.model for v in self.vehicle],
                "num_chosen": [v.num_optional - v.var_optional - v.num_mandatory for v in self.vehicle]})
        car_sol.to_csv("outputVNS/car_sol.csv", index=False)
















