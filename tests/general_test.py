import random


if __name__ == '__main__':
    a = ['a', 'b', 'c']
    count = {
        'a': 0,
        'b': 0,
        'c': 0,
    }
    for i in range(10000):
        count[random.choices(a, [0.01, 0.2, 0.1])[0]] += 1
    print(count)
