import numpy as np
import create_mode


class ListNode:
    def __init__(self, particle=[], next=None):
        self.particle = particle
        self.next = next


class LinkedList:
    def __init__(self):
        self.head = None


    def head_add(self, partical):
        new_node = ListNode(partical)
        new_node.next = self.head
        self.head = new_node


x_bound, y_bound, z_bound = create_mode.create_space()
width = 1
x_num = (x_bound[1] - x_bound[0]) // width
y_num = (y_bound[1] - y_bound[0]) // width
z_num = (y_bound[1] - y_bound[0]) // width
num =  x_num * y_num * z_num
hash_list = [LinkedList() for k in range(num)]


def my_hash(particle_list, len, hash_list):
    global x_num, y_num, z_num, width
    for i in range(len):
        x_index = particle_list[i][3] // width
        y_index = particle_list[i][4] // width
        z_index = particle_list[i][5] // width
        hash_index = z_index * x_num * y_num + y_index * x_num + x_index
        hash_list[hash_index].head_add(particle_list[i])



def find_neighbors(i, particle_list):
    index = np.zeros(10)
    return index