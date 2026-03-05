
import numpy as np
from fontTools.misc.cython import returns

"""
最小路径问题

给定一个三角形 triangle ，找出自顶向下的最小路径和。
每一步只能移动到下一行中相邻的结点上。相邻的结点 在这里指的是 下标 与 上一层结点下标 相同或者等于 上一层结点下标 + 1 的两个结点。
也就是说，如果正位于当前行的下标 i ，那么下一步可以移动到下一行的下标 i 或 i + 1 。
示例 1：
输入：triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
输出：11
解释：如下面简图所示：
   2
  3 4
 6 5 7
4 1 8 3
自顶向下的最小路径和为11（即，2+3+5+1= 11）。
示例 2：
输入：triangle = [[-10]]
输出：-10

"""

triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
depth = triangle.__len__()
res_list = []

def max_path(triangle, i, j):
    if i >= depth:
        return 0

    if j >= triangle[i].__len__():
        return 0


    return  triangle[i][j] + min(max_path(triangle, i+1, j), max_path(triangle, i+1, j+1))


print(max_path(triangle=[[2], [3, 4], [6, 5, 7], [4, 1, 8, 3]], i=0, j=0))