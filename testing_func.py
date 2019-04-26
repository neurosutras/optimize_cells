import numpy as np

hello = {
    1: 200,
    3: 500,
    7: 90
}

test = np.array(hello.items())
print(test)
print(list(test[:,0] / 100.))


# import random
#
# cell1 = [0., 0., 0.]
# #cell2 = [1., -4., 1.]
# cell2 = [3,3,3]
# cells = [cell1, cell2]
# k = 1
#
# def dist(c1, c2):
#     summy = 0
#     for i in range(len(c1)):
#         summy += (c2[i] - c1[i])**2
#     return np.sqrt(summy)
#
# # 0 -> 1
# # 1 -> 0.5
# # 2 -> 0.2
# #3 -> 0.1
#
# def scale_dists(cells):
#     max_dist = 0
#     for cell in cells:
#         maxboi = max(np.abs(cell))
#         if maxboi > max_dist:
#             max_dist = maxboi
#     if max_dist != 0:
#         for cell in cells:
#             for i in range(len(cell)):
#                 cell[i] /= max_dist
#     return cells, max_dist
#
# def connect_or_no(c1, c2):
#     x = dist(c1, c2)
#     #prob = (-(np.e ** x) / (np.e ** x + 1) + 1) * 2
#     prob = -2 / (1 + np.e ** (-k * x)) + 2
#     print(prob)
#     if random.Random().random() <= prob:
#         return True
#     else:
#         return False
#
# _, scale_fac = scale_dists(cells)
# if scale_fac != 0:
#     k = np.sqrt(scale_fac)
# print(k)
# print(scale_fac)
# print(cells)
#
# print([connect_or_no(cell1, cell2) for i in range(10)])
#
#
# #np.random.normal()
