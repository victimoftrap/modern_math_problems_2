import math
import random
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


def main():
    n = 100000
    data = generate_log(size=n, contamination=0.2, primary=lambda: random_log(), secondary=lambda: random_log(scale=10))
    print(sample_characteristics.mean(data))
    print(sample_characteristics.median(data))
    print(sample_characteristics.variance(data))
    print(sample_characteristics.skewness(data))
    print(sample_characteristics.kurtosis(data))
    save_to_isw_file(
        data,
        name='Log(0, 1)x0.8_mixt_scale10_100.dat',
        description='Логистическое(0,1) с засорением 0.2 логистическим (0,10)',
        size=n
    )


if __name__ == '__main__':
    main()
