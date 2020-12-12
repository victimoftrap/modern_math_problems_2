import math
import random
import argparse

import sample_characteristics


def secant_method(f, x0, epsilon, data, scale, delta=None):
    f_0 = f(x0, data, scale, delta)
    f_prime_0 = (f_0 - f(x0 - 0.1, data, scale, delta)) / 0.1
    x_i1 = x0
    x_i2 = x0 - f_0 / f_prime_0
    f_i1 = f_0
    f_i2 = f(x_i2, data, scale, delta)
    while math.fabs(f_i2 - f_i1) > epsilon:
        t = x_i2 - (f_i2 / (f_i2 - f_i1)) * (x_i2 - x_i1)  
        x_i1 = x_i2
        f_i1 = f_i2
        x_i2 = t
        f_i2 = f(t, data, scale, delta)   
    return x_i2
    
    
def estimator_mle_log(theta, data, scale, delta):
    sum = 0
    for y in data:
        t = math.exp((theta - y) / scale)
        sum += (t - 1) / (t + 1)
    return sum
    
    
def estimator_oro_log(theta, data, scale, delta):
    sum = 0
    for y in data:
        t = math.exp((theta - y) / scale)
        sum += math.pow(t, delta) / math.pow((t + 1), 2*delta+1) * (t - 1)
    return sum
    

def random_log(shift=0, scale=1):
    """Моделирование случайной величины логистического распределения.

    :param shift: параматр сдвига, по-умолчанию 0
    :param scale: параметр масштаба, по-умолчанию 1
    :return: значение случайной величины.
    """
    x = random.random()
    return math.log(x / (1 - x)) * scale + shift


def save_to_isw_file(data, size, name, description):
    """Записать датасет в файл формата ISW.

    :param data: набор данных
    :param size: размер набора данных
    :param name: имя файла
    :param description: описание записываемого набора данных
    :return:
    """
    with open(name, 'w') as file:
        file.write(description + '\n')
        file.write(f"0 {size}\n")
        for val in data:
            file.write(f"{val}\n")


def generate_log(size, primary, secondary, contamination):
    """Генерация набора данных.

    :param size: количество значений в выборке
    :param primary: функция чистого распределения
    :param secondary: функция засорённого распределения
    :param contamination: засорение
    :return: набор данных
    """
    res = []
    for i in range(size):
        r = random.random()
        if r <= 1 - contamination:
            res.append(primary())
        else:
            res.append(secondary())
    return res


def main(n, shift, scale, contamination):
    data = generate_log(
        size=n,
        contamination=contamination,
        primary=lambda: random_log(),
        secondary=lambda: random_log(shift=shift, scale=scale)
    )
    mean = sample_characteristics.mean(data)
    print("Среднее:", mean)
    print("Медиана:", sample_characteristics.median(data))
    print("Дисперсия:", sample_characteristics.variance(data))
    print("Коэф. асимметрии:", sample_characteristics.skewness(data))
    print("Коэф. эксцесса:", sample_characteristics.kurtosis(data))
    print("Оценка максимального правдоподобия: ", secant_method(estimator_mle_log, mean, 0.0000001, data, scale))
    print("Обобщенные радикальные оценки (delta=0.1): ", secant_method(estimator_oro_log, mean, 0.0000001, data, scale, 0.1))
    print("Обобщенные радикальные оценки (delta=0.5): ", secant_method(estimator_oro_log, mean, 0.001, data, scale, 0.5))
    #print("Обобщенные радикальные оценки (delta=1): ", secant_method(estimator_oro_log, mean, 0.0000001, data, scale, 1))
    #print("Обобщенные радикальные оценки (delta=0.7): ", secant_method(estimator_oro_log, mean, 0.0000001, data, scale, 0.7))

    save_to_isw_file(
        data,
        name=f'Log(0, 1)x{1 - contamination}_mixt_shift{shift}_scale{scale}_{n}.dat',
        description=f'Логистическое(0,1) с засорением {contamination} логистическим ({shift},{scale})',
        size=n
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Modern math problems, pt.2")
    parser.add_argument('-n', dest='n', type=int, help="объём выборки")
    parser.add_argument('--shift', dest='shift', type=float, help="параметр сдвига")
    parser.add_argument('--scale', dest='scale', type=float, help="параметр масштаба")
    parser.add_argument('-c', '--contamination', dest='contamination', type=float, help="коэффициент засорения")
    parsed_args = parser.parse_args()

    # main(n=100000, shift=0, scale=10)
    main(n=parsed_args.n, shift=parsed_args.shift, scale=parsed_args.scale, contamination=parsed_args.contamination)
