with open("VNS_cpp/VNS.h", "r") as f:
    text = f.read()

# I need to insert `bool success = RuinRebuild(shaken, strength, p);` before `if (success) {`
text = text.replace("int strength = 1 + (nonImprove / 10); \n            \n            if (success) {", "int strength = 1 + (nonImprove / 10); \n            bool success = RuinRebuild(shaken, strength, p);\n            if (success) {")

with open("VNS_cpp/VNS.h", "w") as f:
    f.write(text)

