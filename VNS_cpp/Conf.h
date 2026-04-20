#pragma once

class Config {
public:
    static constexpr int carriage_num = 5;
    static constexpr double eps = 0.0001;
    static constexpr int timelimit = 30;

    static constexpr double A_len = 4300.0;
    static constexpr double B_len = 2000.0;
    static constexpr double C_len = 12400.0;
    static constexpr double D_len = 5000.0;
    static constexpr double E_len = 14900.0;

    static constexpr double A_height_h = 1700.0;
    static constexpr double A_height_m = 1780.0;

    static constexpr double B_height = 2100.0;
    static constexpr double C_height = 2270.0;
    static constexpr double E_height = 2070.0;

    static constexpr double D_height_h = 2070.0;
    static constexpr double D_height_m = 1720.0;

    static constexpr double bottom_len = 2 * A_len + 2 * B_len + C_len;
    static constexpr double top_len = 2 * D_len + E_len;
};