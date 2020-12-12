import math
import random
import argparse

import sample_characteristics


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
        secondary=lambda: random_log(scale=scale)
    )
    print("Среднее:", sample_characteristics.mean(data))
    print("Медиана:", sample_characteristics.median(data))
    print("Дисперсия:", sample_characteristics.variance(data))
    print("Коэф. асимметрии:", sample_characteristics.skewness(data))
    print("Коэф. эксцесса:", sample_characteristics.kurtosis(data))

    save_to_isw_file(
        data,
        name=f'Log(0, 1)x{1 - contamination}_mixt_scale{scale}_{n}.dat',
        description=f'Логистическое(0,1) с засорением {contamination} логистическим (0,{scale})',
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
