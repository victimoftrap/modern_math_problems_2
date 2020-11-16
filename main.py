import math
import random
import stat_analysis


def random_log(shift = 0, scale = 1):
    x = random.random()
    return math.log(x / (1 - x)) * scale + shift


def to_isw_file(data, name, description, size):
    with open(name, 'w') as file:
        file.write(name + '\n')
        file.write(f"0 {size}\n")
        for val in data:
            file.write(f"{val}\n")


def generate_log(size, primary, secondary, contamination):
    res = []
    for i in range(size):
        r = random.random()
        if (r <= 1 - contamination):
            res.append(primary())
        else:
            res.append(secondary())
    return res


def main():
    n = 100000
    data = generate_log(size=n, contamination=0.2, primary=lambda: random_log(), secondary=lambda: random_log(scale=10))
    print(stat_analysis.mean(data))
    print(stat_analysis.median(data))
    print(stat_analysis.variance(data))
    print(stat_analysis.skewness(data))
    print(stat_analysis.kurtosis(data))
    # to_isw_file(data, name='Log(0, 1)x0.8_mixt_scale10_100.dat', description='Логистическое(0,1) с засорением 0.2 логистическим (0,10)', size=n)


if __name__=='__main__':
    main()
