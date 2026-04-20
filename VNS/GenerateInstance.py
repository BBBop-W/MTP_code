import csv
import random

# A list of car brands
brands = ['Toyota', 'Honda', 'Ford', 'Chevrolet', 'BMW', 'Mercedes-Benz', 'Audi', 'Nissan']

# A dictionary of car models by brand
models = {
    'Toyota': ['Corolla', 'Camry', 'RAV4', 'Highlander'],
    'Honda': ['Accord', 'Civic', 'CR-V', 'Pilot'],
    'Ford': ['F-150', 'Escape', 'Explorer', 'Mustang'],
    'Chevrolet': ['Silverado', 'Equinox', 'Traverse', 'Camaro'],
    'BMW': ['i3', 'i8', 'X3', 'X5', 'iX', 'iX M60'],
    'Mercedes-Benz': ['C-Class', 'E-Class', 'S-Class', 'GLC', 'GLE', 'GLS'],
    'Audi': ['A3', 'A4', 'Q5', 'Q7', 'e-tron'],
    'Nissan': ['Altima', 'Sentra', 'Rogue', 'Pathfinder']
}

# Function to generate random car lengths and heights
def generate_dimensions(car_type):
    if car_type in ['Sedan', 'Coupe', 'Hatchback']:
        length = random.randint(4000, 5000)
        height = random.randint(1400, 1500)
        limit1 = random.randint(length, 5000)  # horizontal
        limit2 = random.randint(length, 5000)  # middle
    elif car_type in ['SUV', 'Truck', 'Van']:
        length = random.randint(4500, 6000)
        height = random.randint(1600, 2000)
        limit1 = random.randint(length, 6000)  # horizontal
        limit2 = random.randint(length, 6000)
    else: # Convertible
        length = random.randint(3800, 4800)
        height = random.randint(1200, 1400)
        limit1 = random.randint(length, 4800)  # horizontal
        limit2 = random.randint(length, 4800)
    return length, height, limit1, limit2

# Generate data and write to CSV file
with open('cars.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Brand', 'Model', 'Length', 'Height', 'Optional#', 'Mandatory#', 'Limit_horizontal', 'Limit_middle'])
    data = []
    dict = []
    for i in range(10):
        brand = random.choice(brands)
        model = random.choice(models[brand])
        if (brand, model) in dict:
            continue
        else:
            dict.append((brand, model))


        dict.append((brand, model))
        car_type = random.choice(['Sedan', 'SUV', 'Truck', 'Coupe', 'Hatchback', 'Convertible', 'Van'])
        length, height, limit1, limit2 = generate_dimensions(car_type)
        num_optional = random.randint(0, 20)
        num_mandatory = random.randint(0, num_optional)
        data.append([brand, model, length, height, num_optional, num_mandatory, limit1, limit2])
    data_sorted = sorted(data, key=lambda x: (x[0], x[1])) # Sort by brand, then model
    writer.writerows(data_sorted)
