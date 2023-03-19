import numpy as np
import copy
import queue

x = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
y = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 0, 8]])

target = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 0, 8]])


class Matrix(object):
    def __init__(self, init_state: np.array):
        self.matrix_ = init_state
        # print(np.argwhere(self.matrix_ == 0))
        self.zero_x_ = np.argwhere(self.matrix_ == 0)[0][0]
        self.zero_y_ = np.argwhere(self.matrix_ == 0)[0][1]
        self.depth = 0

    def __lt__(self, other):  # 反转大小比较结果，用于实现小顶堆
        return self.f_n() < other.f_n()

    def move_up(self):
        if self.zero_x_ == 0:
            return False
        tmp = self.matrix_[self.zero_x_ - 1][self.zero_y_]
        self.matrix_[self.zero_x_ - 1][self.zero_y_] = 0
        self.matrix_[self.zero_x_][self.zero_y_] = tmp
        self.zero_x_ -= 1
        return True

    def move_down(self):
        if self.zero_x_ == 2:
            return False
        tmp = self.matrix_[self.zero_x_ + 1][self.zero_y_]
        self.matrix_[self.zero_x_ + 1][self.zero_y_] = 0
        self.matrix_[self.zero_x_][self.zero_y_] = tmp
        self.zero_x_ += 1
        return True

    def move_left(self):
        if self.zero_y_ == 0:
            return False
        tmp = self.matrix_[self.zero_x_][self.zero_y_ - 1]
        self.matrix_[self.zero_x_][self.zero_y_ - 1] = 0
        self.matrix_[self.zero_x_][self.zero_y_] = tmp
        self.zero_y_ -= 1
        return True

    def move_right(self):
        if self.zero_y_ == 2:
            return False
        tmp = self.matrix_[self.zero_x_ - 1][self.zero_y_]
        self.matrix_[self.zero_x_][self.zero_y_ + 1] = 0
        self.matrix_[self.zero_x_][self.zero_y_] = tmp
        self.zero_y_ += 1
        return True

    def w_n(self):
        return 9 - np.sum(self.matrix_ == target)

    def d_n(self):
        return self.depth

    def f_n(self):
        return self.w_n() + self.d_n()

    def is_target(self):
        return self.w_n() == 0

    def __repr__(self):
        print(self.matrix_)
        return "depth: " + str(self.depth) + " f_n: " + str(self.f_n()) + "\n"


def move(matrix_: Matrix, pq_: queue.PriorityQueue):
    matrix_up = copy.deepcopy(matrix_)
    if matrix_up.move_up():
        matrix_up.depth += 1
        pq_.put(matrix_up)
    matrix_down = copy.deepcopy(matrix_)
    if matrix_down.move_down():
        matrix_down.depth += 1
        pq_.put(matrix_down)
    matrix_left = copy.deepcopy(matrix_)
    if matrix_left.move_left():
        matrix_left.depth += 1
        pq_.put(matrix_left)
    matrix_right = copy.deepcopy(matrix_)
    if matrix_right.move_right():
        matrix_right.depth += 1
        pq_.put(matrix_right)
    return pq_


matrix = Matrix(np.array([[1, 3, 6],
                          [4, 2, 8],
                          [7, 5, 0]]))

pq = queue.PriorityQueue()
pq.put(matrix)

while not pq.empty():
    matrix_get = pq.get()
    if matrix_get.is_target():
        print("target found! \n")
        print(matrix_get)
        break

    pq = move(matrix_get, pq)
    print("queue size: ", pq.qsize())
