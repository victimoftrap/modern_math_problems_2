import math
import random
import argparse
from functools import partial

import numpy
from scipy import optimize

import sample_characteristics


def monte_carlo(m, n, contamination, shift, scale):
    """Исследование оценок методом Монте-Карло.

    :param m: число испытаний
    :param n: объём выборки
    :param contamination: засорение
    :param shift: параматр сдвига
    :param scale: параметр масштаба
    :return:
    """
    omp = []
    oro_1 = []
    oro_5 = []
    oro_10 = []

    oro_1_res = []
    oro_5_res = []
    oro_10_res = []
    for i in range(m):
        data = generate_log(
            size=n,
            contamination=contamination,
            primary=lambda: random_log(),
            secondary=lambda: random_log(shift=shift, scale=scale)
        )
        omp.append(secant_method(estimator_mle_log, 0, 0.0000001, data, scale) ** 2)
        oro_1.append(optimize.root_scalar(partial(estimator_oro_log, data, scale, 0.1), x0=0,
                                          fprime=partial(estimator_oro_log_prime, data, scale, 0.1),
                                          method='newton').root ** 2)
        oro_1_res.append(estimator_oro_log(data, scale, 0.1, oro_1[i]))

        oro_5.append(optimize.root_scalar(partial(estimator_oro_log, data, scale, 0.5), x0=0,
                                          fprime=partial(estimator_oro_log_prime, data, scale, 0.5),
                                          method='newton').root ** 2)

        oro_5_res.append(estimator_oro_log(data, scale, 0.5, oro_5[i]))
        oro_10.append(optimize.root_scalar(partial(estimator_oro_log, data, scale, 1), x0=0,
                                           fprime=partial(estimator_oro_log_prime, data, scale, 1),
                                           method='newton').root ** 2)
        oro_10_res.append(estimator_oro_log(data, scale, 1, oro_10[i]))

    # hagrid = {oro_10[i]: oro_10_res[i] for i in range(len(oro_10))}
    # sorted_hagrid = collections.OrderedDict(sorted(hagrid.items()))
    # plt.plot(sorted_hagrid.keys(), sorted_hagrid.values())
    # plt.show()
    print(f"Критерий качества ОМП: {numpy.mean(omp)}")
    print(f"Критерий качества ОРО 0.1: {numpy.mean(oro_1)}")
    print(f"Критерий качества ОРО 0.5: {numpy.mean(oro_5)}")
    print(f"Критерий качества ОРО 1: {numpy.mean(oro_10)}")


def secant_method(f, x0, epsilon, data, scale, delta=None):
    f_0 = f(data, scale, delta, x0)
    f_prime_0 = (f_0 - f(data=data, scale=scale, delta=delta, theta=x0 - 0.1)) / 0.1
    x_i1 = x0
    x_i2 = x0 - f_0 / f_prime_0
    f_i1 = f_0
    f_i2 = f(data=data, scale=scale, delta=delta, theta=x_i2)
    while math.fabs(x_i2 - x_i1) > epsilon:
        t = x_i2 - (f_i2 / (f_i2 - f_i1)) * (x_i2 - x_i1)
        x_i1 = x_i2
        f_i1 = f_i2
        x_i2 = t
        f_i2 = f(data=data, scale=scale, delta=delta, theta=t)
    return x_i2


def estimator_mle_log(data, scale, delta, theta):
    """Оценочная функция оценки максимальног оправдоподобия.

    :param data: набор данных
    :param scale: параметр масштаба
    :param delta: регулирование степени робастности
    :param theta: приближение
    :return:
    """
    s = 0
    for y in data:
        t = math.exp((theta - y) / scale)
        s += (t - 1) / (t + 1)
    return s


def estimator_oro_log(data, scale, delta, theta):
    """Оценочная функция обобщённой радикальной оценки.

    :param data: набор данных
    :param scale: параметр масштаба
    :param delta: регулирование степени робастности
    :param theta: приближение
    :return:
    """
    s = 0
    for y in data:
        t = math.exp((theta - y) / scale)
        addition = math.pow(t, delta) / math.pow((t + 1), 2 * delta + 1) * (t - 1)
        s += addition
    return s


def estimator_oro_log_prime(data, scale, delta, theta):
    """Оценочная функция обобщённой радикальной оценки.

    :param data: набор данных
    :param scale: параметр масштаба
    :param delta: регулирование степени робастности
    :param theta: приближение
    :return:
    """
    s = 0
    for y in data:
        t = math.exp((theta - y) / scale)
        addition = (2 * delta + 1) * (t - 1) * math.pow(t, delta + 1) * math.pow(t + 1, 2 * (-delta) - 2) \
            - math.pow(t, delta + 1) * math.pow(t + 1, 2 * (-delta) - 1) \
            - delta * (t - 1) * math.pow(t, delta) * math.pow(t + 1, 2 * (-delta) - 1)
        s += addition
    return s / (-scale)


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
    monte_carlo(1000, n, contamination, shift, scale)


def main2(n, shift, scale, contamination):
    data = generate_log(
        size=n,
        contamination=contamination,
        primary=lambda: random_log(),
        secondary=lambda: random_log(shift=shift, scale=scale)
    )
    mean = sample_characteristics.mean(data)
    median = sample_characteristics.median(data)
    print("Среднее:", mean)
    print("Медиана:", median)
    print("Дисперсия:", sample_characteristics.variance(data))
    print("Коэф. асимметрии:", sample_characteristics.skewness(data))
    print("Коэф. эксцесса:", sample_characteristics.kurtosis(data))
    mle = secant_method(estimator_mle_log, mean, 0.0000001, data, scale)
    print("Оценка максимального правдоподобия: ", mle)

    print("\nОбобщенные радикальные оценки (delta=0.1): ",
          optimize.root_scalar(partial(estimator_oro_log, data, scale, 0.1), x0=0,
                               fprime=partial(estimator_oro_log_prime, data, scale, 0.1), method='newton').root)
    print("\nОбобщенные радикальные оценки (delta=0.5): ",
          optimize.root_scalar(partial(estimator_oro_log, data, scale, 0.5), x0=0,
                               fprime=partial(estimator_oro_log_prime, data, scale, 0.5), method='newton').root)
    print("\nОбобщенные радикальные оценки (delta=1): ",
          optimize.root_scalar(partial(estimator_oro_log, data, scale, 1), x0=0,
                               fprime=partial(estimator_oro_log_prime, data, scale, 1), method='newton').root)

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

    main(n=parsed_args.n, shift=parsed_args.shift, scale=parsed_args.scale, contamination=parsed_args.contamination)
