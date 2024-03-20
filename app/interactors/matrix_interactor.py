from random import random


class Matrix:
    def __init__(self,
                 rows: int,
                 cols: int,):
        self.rows = rows
        self.cols = cols
        self.data = []

        for row in range(rows):
            arr = []
            for col in range(cols):
                arr.append(0)
            self.data.append(arr)

    def print(self):
        print('==============================')
        for index, row in enumerate(self.data):
            print(f'{row}')

    def map(self, func):
        self.data = [
            [func(col, r, c) for c, col in enumerate(row)] for r, row in enumerate(self.data)
        ]

    @staticmethod
    def static_map(matrix_a, func):
        matrix_a.data = [
            [func(col, r, c) for c, col in enumerate(row)] for r, row in enumerate(matrix_a.data)
        ]
        return matrix_a

    @staticmethod
    def array_to_matrix(arr: list):
        matrix = Matrix(len(arr), 1)

        def func(elm, i, j):
            return arr[i]

        matrix.map(func)
        return matrix

    @staticmethod
    def matrix_to_array(matrix):
        arr = []

        def func(elm, i, j):
            return arr.append(elm)

        matrix.map(func)
        return arr

    def randomize(self):
        def func(elm, index_1, index_2):
            return random() * 2 - 1

        self.map(func)

    @staticmethod
    def transpose(matrix_a):
        matrix = Matrix(matrix_a.cols, matrix_a.rows)

        def func(elm, index_1, index_2):
            return matrix_a.data[index_2][index_1]

        matrix.map(func)
        return matrix

    @staticmethod
    def scaled_multiply(matrix_a, scaled):
        matrix = Matrix(matrix_a.rows, matrix_a.cols)

        def sum_arrays(elm, index_1, index_2):
            return matrix_a.data[index_1][index_2] * scaled

        matrix.map(sum_arrays)
        return matrix

    @staticmethod
    def hadamard(matrix_a, matrix_b):
        matrix = Matrix(matrix_a.rows, matrix_a.cols)

        def func(elm, index_1, index_2):
            return matrix_a.data[index_1][index_2] * matrix_b.data[index_1][index_2]

        matrix.map(func)
        return matrix

    @staticmethod
    def subtract(matrix_a, matrix_b):
        matrix = Matrix(matrix_a.rows, matrix_a.cols)

        def func(elm, index_1, index_2):
            return matrix_a.data[index_1][index_2] - matrix_b.data[index_1][index_2]

        matrix.map(func)
        return matrix

    @staticmethod
    def add(matrix_a, matrix_b):
        matrix = Matrix(matrix_a.rows, matrix_a.cols)

        def func(elm, index_1, index_2):
            return matrix_a.data[index_1][index_2] + matrix_b.data[index_1][index_2]

        matrix.map(func)
        return matrix

    @staticmethod
    def multiply(matrix_a, matrix_b):
        matrix = Matrix(matrix_a.rows, matrix_b.cols)

        def func(elm, i, j):
            sum_all_multiply = 0

            for k in range(matrix_a.cols):
                sum_all_multiply += matrix_a.data[i][k] * matrix_b.data[k][j]

            return sum_all_multiply

        matrix.map(func)

        return matrix

