with open("VNS_cpp/Conf.h", "r") as f:
    text = f.read()

if "get_generator" not in text:
    text = "#include <random>\ninline std::mt19937& get_generator() { static thread_local std::mt19937 gen(42); return gen; }\n" + text
    with open("VNS_cpp/Conf.h", "w") as f:
        f.write(text)

import glob
for file in glob.glob("VNS_cpp/*.h"):
    with open(file, "r") as f:
        content = f.read()
    
    content = content.replace("std::random_device rd;\n        std::mt19937 g(rd());", "std::mt19937& g = get_generator();")
    content = content.replace("std::random_device rd;\n    std::mt19937 generator(rd());", "std::mt19937& generator = get_generator();")
    
    with open(file, "w") as f:
        f.write(content)
