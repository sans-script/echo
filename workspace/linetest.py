import sys


def parse(line):
    return [int(x) for x in line.split(',')]


def total(values):
    return sum(values)


if __name__ == '__main__':
    print(total(parse(sys.argv[1])))
