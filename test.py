#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import json

# x = [0, 1, 2, 3, 4]
# y = [ [0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [9, 8, 7, 6, 5] , [1, 5, 6, 2, 1], [1,3,4,5,7]]

# x = np.array(x)
# y = np.array(y)
# # y = np.transpose(y)
# labels = ['foo', 'bar', 'baz', 'lolo', 'lala']

# for y_arr, label in zip (y, labels):
#     plt.plot(x, y_arr.transpose(), label=label)

# plt.legend()
# plt.show()

with open('motion_imitation/data/motions/dog_pace.txt') as f:
    data = json.load(f)

frames = data['Frames']
frames = np.array(frames)
position = np.transpose(frames)[0:3]

x = np.linspace(0,position.shape[1], position.shape[1])
labels = ['x', 'y', 'z']

for y, label in zip(position, labels):
    plt.plot(x,y,label=label)
plt.legend()
plt.show()