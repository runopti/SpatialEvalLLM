import math
import numpy as np
import random

import json, jsonlines, datetime, string
from collections import Counter

def answer2jsonl(answers, model_outputs, questions, out_file):
    # Confirm we have answers for all questions
    assert len(answers) == len(questions)
    outputs = []
    for q_idx in range(len(answers)):
        output = {"question": questions[q_idx]["question"], "answer": questions[q_idx]["answer"], "prediction": answers[q_idx], "explanation": model_outputs[q_idx]}
        outputs.append(output)
    with jsonlines.open(out_file, mode='w') as fout:
        fout.write_all(outputs)
        
def read_jsonl(in_file):
    questions = []
    with open(in_file) as fin:
        for line in fin:
            question = json.loads(line)
            questions.append(question)
    return questions


class Vec2D(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vec2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec2D(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        return self.x*other.x + self.y*other.y

    def __abs__(self):
        return math.sqrt(self.x**2 + self.y**2)

    def __eq__(self, other):
        return np.allclose(self.x, other.x) and \
               np.allclose(self.y, other.y)

    def __str__(self):
        return '(%g, %g)' % (self.x, self.y)

    def __ne__(self, other):
        return not self.__eq__(other)  # reuse __eq__

class Vec3D(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Vec3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vec3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        return self.x*other.x + self.y*other.y + self.z * other.z

    def __abs__(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def __eq__(self, other):
        return np.allclose(self.x, other.x) and \
               np.allclose(self.y, other.y) and \
               np.allclose(self.z, other.z)

    def __str__(self):
        return '(%g, %g, %g)' % (self.x, self.y, self.z)

    def __ne__(self, other):
        return not self.__eq__(other)  # reuse __eq__


class HexGrid():
    def __init__(self, radius):
        deltas = [[1,0,-1],[0,1,-1],[-1,1,0],[-1,0,1],[0,-1,1],[1,-1,0]]
        self.radius = radius
        self.tiles = {(0, 0, 0): "X"}
        for r in range(radius + 1):
            a = 0
            b = -r
            c = +r
            for j in range(6):
                num_of_hexas_in_edge = r
                for i in range(num_of_hexas_in_edge):
                    a = a+deltas[j][0]
                    b = b+deltas[j][1]
                    c = c+deltas[j][2]           
                    self.tiles[a,b,c] = "X"

    def show(self):
        l = []
        for y in range(20):
            l.append([])
            for x in range(60):
                l[y].append(".")
        for (a,b,c), tile in self.tiles.items():
            l[self.radius-1-b][a-c+(2*(self.radius-1))] = self.tiles[a,b,c]
        mapString = ""
        for y in range(len(l)):
            for x in range(len(l[y])):
                mapString += l[y][x]
            mapString += "\n"
        print(mapString)


if __name__ == "__main__":
    pos = Vec2D(1,2) + Vec2D(2,3)
    pos2 = Vec3D(1,2,3) + Vec3D(2,3,1)
    gg = HexGrid(radius=1)

