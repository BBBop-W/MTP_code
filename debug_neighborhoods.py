with open("VNS_cpp/Neighborhoods.h") as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if "InterRelocateRandom" in l or "InterSwapRandom" in l or "InterOptRandom" in l:
        print("".join(lines[i:i+40]))
