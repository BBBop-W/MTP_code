import sys

with open("VNS_cpp/Feasibility.h", "r") as f:
    text = f.read()

text = text.replace("Config.A_height", "Config::A_height")
text = text.replace("Config.A_len", "Config::A_len")
text = text.replace("Config.B_height", "Config::B_height")
text = text.replace("Config.B_len", "Config::B_len")
text = text.replace("Config.C_height", "Config::C_height")
text = text.replace("Config.C_len", "Config::C_len")
text = text.replace("Config.D_height", "Config::D_height")
text = text.replace("Config.D_len", "Config::D_len")
text = text.replace("Config.E_height", "Config::E_height")
text = text.replace("Config.E_len", "Config::E_len")

with open("VNS_cpp/Feasibility.h", "w") as f:
    f.write(text)
