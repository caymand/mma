import colorama
import numpy as np
from colorama import Back

core_matrix_X = 8

X = 4
Y = 8

colors = {0: Back.BLUE, 1: Back.GREEN, 2: Back.YELLOW, 3: Back.RED, 4: Back.CYAN, 5: Back.MAGENTA, 6: Back.WHITE, 7: Back.BLACK}

def print_cell(x, y):
    print(colors[y] + f"({x * core_matrix_X:02}...{(x+1) * core_matrix_X - 1:02}, {y})", end=Back.RESET + "\t")

def print_newline():
    print(Back.RESET + "\n")

# Original
for y in range(Y):
    for x in range(X):
        print_cell(x, y)

    print(Back.RESET + "\n")

print("\n\n\n")

# Option 2
def global_shared_map(x, y):
    new_y = x % X
    new_x = y ^ new_y

    return new_x, new_y

shared_global_map = {}

for y in range(Y):
    for x in range(X):
        new_x, new_y = global_shared_map(x, y)

        shared_global_map[(new_x, new_y)] = (x, y)


for y in range(X):
    for x in range(Y):
        src_x, src_y = shared_global_map[(x, y)]
        print_cell(src_x, src_y)

    print_newline()

print("\n\n\n")


for y in range(X):
    for x in range(Y):
        src_x, src_y = shared_global_map[(x, y)]
        print_cell(src_x, src_y ^ src_x)
    print_newline()

