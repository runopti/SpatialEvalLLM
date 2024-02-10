import math
import json
import os

import networkx as nx 
import inflect

import numpy as np
import argparse

from utils import Vec2D, HexGrid, Vec3D

parser = argparse.ArgumentParser()

article = inflect.engine()

def generate_map(maptype, size, n_obj): 
    arr = np.arange(n_obj)
    np.random.shuffle(arr) 
    if maptype in {'square', 'rhombus'}:
        mat = np.empty([size, size], dtype=int)
        for i in range(size):
            for j in range(size):
                mat[i,j] = arr[i*size + j]
    elif maptype in {'rectangle', 'rectanglenavonly'}:
        mat = np.empty([size[0], size[1]], dtype=int)
        for i in range(size[0]):
            for j in range(size[1]):
                mat[i,j] = arr[i*size[1] + j]
    elif maptype == 'ring':
        mat = np.empty([size], dtype=int)
        for i in range(size):
            mat[i] = arr[i]
    elif maptype == 'hex_deprecated':
        mat = HexGrid(size).tiles
        for i, key in enumerate(mat.keys()):
            mat[key] = arr[i]
    elif maptype == 'triangle':
        mat = -2 * np.ones([2*size+1, size+1], dtype=int)
        count = 0
        for j in range(size+1):
            for i in range(j, 2*(size+1)-j, 2):
                #print(i,j)
                mat[i,j] = arr[count]
                count += 1
    elif maptype == 'hexagon':
        mat = -2 * np.ones([(2*size)+2, (2*size)+2], dtype=int)
        count = 0
        for j in range(2*size+2):
            for i in range((j+1)//2, 2*size+3-(j+1)//2, 2):
                if (i,j) == (0,0) or (i,j) == (2*size+2,0):
                    continue
                else:
                    #print(i,j)
                    mat[i,j] = arr[count]
                    count += 1
    else:
        raise NotImplementedError(f'maptype {maptype} not supported')
    return mat


def sample_path_ring(mat, size):
    # Sample a sequence of actions that traverses the ring.
    # The sequence of actions is a list of [clockwise or counter-clockwise, how many steps to take]. 
    start = 0 
    commands = [i for i in range(1, size+1)] + [-1*i for i in range(1, size+1)]
    p_dist = [1.0/(2*size) for i in range(2*size)]
    prev = start
    path = [start]
    mat[start] = -1
    actions = []
    while True:
        command = np.random.choice(commands, p=p_dist)
        pos = (prev + command) % size
        path.append(pos)
        actions.append(command)
        if mat[pos] == -1:
            return path, actions
        else:
            mat[pos] = -1
            prev = pos


def sample_path_hexagon(mat, size):
    # Sample a sequence of actions that traverses the hex grid until it reaches a visited node.
    # There are 3 possible actions if len(actions) is even: "up", "down-left", and "down-right".
    # There are 3 possible actions if len(actions) is odd: "up-right", "up-left", and "down".
    def is_valid(pos):
        return (pos.x >= 0 and pos.x < 2*size+2 and pos.y >= 0 and pos.y < 2*size+2 and mat[pos.x][pos.y] != -2)
    
    def get_pdist(command):
        if command.x == 0 and command.y == 1: # up
            return [0.495, 0.495, 0.01] # lower prob of going down
        elif command.x == -1 and command.y == -1: # down-left
            return [0.01, 0.495, 0.495] # lower prob of going up-right
        elif command.x == 1 and command.y == -1: # down-right
            return [0.495, 0.01, 0.495] # lower prob of going up-left
        elif command.x == 1 and command.y == 1: # up-right
            return [0.495, 0.01, 0.495] # lower prob of going down-left
        elif command.x == -1 and command.y == 1: # up-left
            return [0.495, 0.495, 0.01] # lower prob of going down-right
        elif command.x == 0 and command.y == -1: # down
            return [0.01, 0.495, 0.495] # lower prob of going up
        else:
            raise NotImplementedError()

    start = np.random.choice([Vec2D(1,1), Vec2D(2*size+1,1), Vec2D(size+1,2*size+1)])
    commands_even = [Vec2D(0,1), Vec2D(-1,-1), Vec2D(1,-1)]
    commands_odd = [Vec2D(1,1), Vec2D(-1,1), Vec2D(0,-1)]
    p_dist = [1.0/3 for i in range(3)]
    prev = start
    path = [start]
    mat[start.x][start.y] = -1
    actions = []
    while True:
        if len(actions) % 2 == 0:
            command = np.random.choice(commands_even, p=p_dist)
        else:
            command = np.random.choice(commands_odd, p=p_dist)
        pos = prev + command
        #print(f"prev: {prev.x}, {prev.y}")
        #print(f"pos: {pos.x}, {pos.y}")
        if is_valid(pos):
            path.append(pos)
            actions.append(command)
            p_dist = get_pdist(command)
            if mat[pos.x][pos.y] == -1:
                return path, actions
            else:
                mat[pos.x][pos.y] = -1
                prev = pos
 

def sample_path_triangle(mat, size):
    # Sample a sequence of actions that traverses the triangle grid until it reaches a visited node.
    # There are 6 possible actions: "right", "up-right", "up-left", "left", "down-left", and "down-right".
    def is_valid(pos):
        return (pos.x >= 0 and pos.x < 2*size+1 and pos.y >= 0 and pos.y < size+1 and mat[pos.x][pos.y] != -2)
    def get_pdist(command):
        if command.x == 2 and command.y == 0: #right
            return [0.198, 0.198, 0.198, 0.01, 0.198, 0.198] # lower prob of going left
        elif command.x == 1 and command.y == 1: #up-right
            return [0.198, 0.198, 0.198, 0.198, 0.01, 0.198] # lower prob of going down-left
        elif command.x == -1 and command.y == 1: #up-left
            return [0.198, 0.198, 0.198, 0.198, 0.198, 0.01] # lower prob of going down-right
        elif command.x == -2 and command.y == 0: #left
            return [0.01, 0.198, 0.198, 0.198, 0.198, 0.198] # lower prob of going right
        elif command.x == -1 and command.y == -1: #down-left
            return [0.198, 0.01, 0.198, 0.198, 0.198, 0.198] # lower prob of going up-right
        elif command.x == 1 and command.y == -1: #down-right
            return [0.198, 0.198, 0.01, 0.198, 0.198, 0.198] # lower prob of going up-left
        else:
            raise NotImplementedError()

    start = np.random.choice([Vec2D(0, 0), Vec2D(size, size), Vec2D(2*size, 0)])
    commands = [Vec2D(2,0), Vec2D(1,1), Vec2D(-1,1), Vec2D(-2,0), Vec2D(-1,-1), Vec2D(1,-1)]
    p_dist = [1.0/6 for i in range(6)]
    prev = start
    path = [start]
    mat[start.x][start.y] = -1
    actions = []
    while True:
        command = np.random.choice(commands, p=p_dist)
        pos = prev + command
        if prev.x == 3 and prev.y == 0:
            import ipdb; ipdb.set_trace()
        #print(f"prev: {prev.x}, {prev.y}")
        #print(f"pos: {pos.x}, {pos.y}")
        if is_valid(pos):
            path.append(pos)
            actions.append(command)
            p_dist = get_pdist(command)
            if mat[pos.x][pos.y] == -1:
                return path, actions
            else:
                mat[pos.x][pos.y] = -1
                prev = pos

def sample_path_rectangle(all_paths, all_actions):
    random_index = np.random.randint(0, len(all_paths))
    path = all_paths[random_index]
    act = all_actions[random_index]
    path_new = []
    act_new = []
    for i in range(len(path)):
        path_new.append(Vec2D(path[i][0], path[i][1]))
        if i < len(act):
            act_new.append(Vec2D(act[i][0], act[i][1]))
    return path_new, act_new
    

def sample_path_square(mat, size):
    def is_valid(pos):
        return (pos.x >= 0 and pos.x < size and pos.y >= 0 and pos.y < size)

    def visited(pos):
        return mat[pos.x, pos.y] == -1

    def get_pdist(command):
        if command.x == 0 and command.y == 1:
            return [0.33, 0.33, 0.01, 0.33]
        elif command.x == 1 and command.y == 0:
            return [0.33, 0.33, 0.33, 0.01]
        elif command.x == 0 and command.y == -1:
            return [0.01, 0.33, 0.33, 0.33]
        elif command.x == -1 and command.y == 0:
            return [0.33, 0.01, 0.33, 0.33]
        else:
            raise NotImplementedError()


    start = np.random.choice([Vec2D(0,0), Vec2D(0, size-1),
                              Vec2D(size-1, 0), Vec2D(size-1, size-1)])
    commands = [Vec2D(0, 1), Vec2D(1, 0), Vec2D(0, -1), Vec2D(-1, 0)]
    p_dist = [0.25, 0.25, 0.25, 0.25]
    prev = start
    path = [start]
    mat[start.x, start.y] = -1
    actions = []
    while True:
        command = np.random.choice(commands, p=p_dist)
        pos = prev + command 
        if is_valid(pos):
            path.append(pos)
            actions.append(command)
            p_dist = get_pdist(command)
            if visited(pos):
                return path, actions
            else:
                mat[pos.x, pos.y] = -1
                prev = pos


def get_init_loc_to_string(mattype, size):
    if mattype == 'square':
        return {
            (0,0): "the bottom left corner",
            (0, size-1): "the top left corner",
            (size-1, 0): "the bottom right corner",
            (size-1, size-1): "the top right corner",
        }
    elif mattype == 'rhombus':
        return {
            (0,0): "the left corner",
            (0, size-1): "the top corner",
            (size-1, 0): "the bottom corner",
            (size-1, size-1): "the right corner",
        }
    elif mattype in {'rectangle', 'rectanglenavonly'}:
        return {
            (0,0): "the bottom left corner",
            (0, size[1]-1): "the top left corner",
            (size[0]-1, 0): "the bottom right corner",
            (size[0]-1, size[1]-1): "the top right corner",
        }
    elif mattype == 'ring':
        return ""
    elif mattype == 'hexagon':
        return {
            (1,1): "the bottom left corner",
            (2*size+1, 1): "the bottom right corner",
            (size+1, 2*size+1): "the top corner",
        }
    elif mattype == 'triangle':
        return {
            (0,0): "the bottom left corner",
            (2*size, 0): "the bottom right corner",
            (size, size): "the top corner",
        }
    else:
        raise NotImplementedError()


def init_cond(obj_name, pos, size, maptype, mat):
    init_loc_to_string = get_init_loc_to_string(maptype, size)

    if maptype == 'square':
        string = "You have been given a "+str(size)+" by "+str(size)+" square grid. "
        string += "Starting from a vertex, you will move along the edges of the grid. "
        string += "Initially, you are positioned at "
        string += init_loc_to_string[(pos.x, pos.y)]
        string += " of the grid, "
    elif maptype == 'rhombus':
        string = f"You have been given a {size} by {size} pointy-topped regular rhombus tile map. "
        string += "Starting from a vertex, you will move along the edges of the grid. "
        string += "Initially, you are positioned at "
        string += init_loc_to_string[(pos.x, pos.y)]
        string += " of the grid, "
    elif maptype in {'rectangle', 'rectanglenavonly'}: 
        string = "You have been given a rectangular grid of unknown height and width. "
        string += "Starting from a vertex, you will move along the edges of the grid. "
        string += "Initially, you are positioned at "
        string += init_loc_to_string[(pos.x, pos.y)]
        if maptype == "rectanglenavonly":
            string += " of the grid. "
        else:
            string += " of the grid, "
    elif maptype == 'tree':
        string += "a tree with " + str(size) + " nodes ,"
    
    elif maptype == 'ring':
        string = f"You have been given a circular grid consisting of {size} connected dots. "
        string += "Starting from a vertex, you will move along the edges of the circular grid. "
        string += "Initially, you are positioned on the dot that's located at the top of the grid, "
    elif maptype == 'hexagon':
        string = f"You have been given a pointy-topped regular hexagonal tile map consisting of {size}" 
        if size == 1:
            string += " tile. "
        elif size == 2:
            string += " rows, where the first row has one tile and the second row has two tiles. "
        else:
            string += " rows, where the first row has one tile, the second row has two tiles, and so on. "
        string += "Starting from a vertex, you will move along the edges of "
        string += "the tile. " if size == 1 else "these tiles. " 
        string += "Initially, you are positioned at "
        string += init_loc_to_string[(pos.x, pos.y)]
        string += " of the map, "
    elif maptype == 'triangle':
        string = f"You have been given an equilateral triangular tile map consisting of {size}"
        if size == 1:
            string += " tile. "
        elif size == 2:
            string += " rows, where the first row has one tile and the second row has three tiles. "
        else:
            string += " rows, where the first row has one tile, the second row has three tiles, and so on, so that the i th row has 2*i-1 tiles. "
        #string += "Starting from a vertex, you will move along the edges of these tiles. Initially, you are positioned at "
        string += "Starting from a vertex, you will move along the edges of "
        string += "the tile. " if size == 1 else "these tiles. " 
        string += "Initially, you are positioned at "
        string += init_loc_to_string[(pos.x, pos.y)]
        string += " of the map, "
    else:
        raise NotImplementedError()
    if maptype == "rectanglenavonly":
        return string
    else:
        string += "where you find " + article.a(obj_name) + ". "
        return string

def get_act_to_string(mattype):
    if mattype == 'square':
        act_to_string = {
            (0,1): "up", 
            (1,0): "right",
            (0,-1): "down", 
            (-1,0): "left",
        }
        return act_to_string
    elif mattype == 'rhombus':
        act_to_string = {
            (0,1): "up-right", 
            (1,0): "down-right",
            (0,-1): "down-left", 
            (-1,0): "up-left",
        }
        return act_to_string
    elif mattype in {'rectangle', 'rectanglenavonly'}:
        act_to_string = {
            (0,1): "up", 
            (1,0): "right",
            (0,-1): "down", 
            (-1,0): "left",
        }
        return act_to_string
    elif mattype == 'ring':
        act_to_string = {
            1: "clockwise",
            -1: "counter-clockwise",
        }
        return act_to_string
    elif mattype == 'hex_deprecated':
        act_to_string = {
            "northeast": Vec3D(0, -1, 1),
            "northwest": Vec3D(-1, 0, 1),
            "southeast": Vec3D(1, 0, -1),
            "southwest": Vec3D(0, 1, -1),
            "east": Vec3D(1, -1, 0),
            "west": Vec3D(-1, 1, 0),
        }
    elif mattype == "hexagon":
        return {
            (1,1): "up-right",
            (-1, 1): "up-left",
            (0,-1): "down",
            (1,-1): "down-right",
            (-1,-1): "down-left",
            (0,1): "up",
        }
    elif mattype == 'triangle':
        return  {
            (2,0) : "right",
            (1,1) : "up-right",
            (-1,1) : "up-left",
            (-2,0) : "left",
            (-1,-1) : "down-left",
            (1,-1) : "down-right",
        }
    else:
        raise NotImplementedError()

def get_description(act, obj_name, maptype, is_last):
    string = "You move "
    act_to_string = get_act_to_string(maptype)
    if maptype == "square":
        string += act_to_string[(act.x, act.y)] + " by one step"
        if is_last:
            string += ". What will you find?"
        else:
            string += ", where you find " + article.a(obj_name) + ". "
    elif maptype == "rhombus":
        string += act_to_string[(act.x, act.y)] + " by one step"
        if is_last:
            string += ". What will you find?"
        else:
            string += ", where you find " + article.a(obj_name) + ". "
    elif maptype == "rectangle":
        if is_last:
            string = "It is at this point that you realize you have visited all locations on the grid. What is the height and width of the grid?"
        else:
            string += act_to_string[(act.x, act.y)] + " by one step"
            string += ", where you find " + article.a(obj_name) + ". "
    elif maptype == "rectanglenavonly":
        if is_last:
            string = "It is at this point that you realize you have visited all locations on the grid. What is the height and width of the grid?"
        else:
            string += act_to_string[(act.x, act.y)] + " by one step. "
    elif maptype == "ring":
        string += f"around the ring by {str(np.abs(act))}"
        string += " step " if np.abs(act) == 1 else " steps "
        string +="in a "+ act_to_string[np.sign(act)] +" direction"
        if is_last:
            string += ". What will you find?"
        else:
            string += ", where you find " + article.a(obj_name) + ". "
    elif maptype == "triangle":
        string += act_to_string[(act.x, act.y)] + " by one step"
        if is_last:
            string += ". What will you find?"
        else:
            string += ", where you find " + article.a(obj_name) + ". "
    elif maptype == "hexagon":
        string += act_to_string[(act.x, act.y)] + " by one step"
        if is_last:
            string += ". What will you find?"
        else:
            string += ", where you find " + article.a(obj_name) + ". "
    else:
        raise NotImplementedError()
    return string

def get_obj_name(mat, pos, obj_names, maptype):
    if maptype == 'square':
        return obj_names[mat[pos.x][pos.y]]
    elif maptype == 'rhombus':
        return obj_names[mat[pos.x][pos.y]]
    elif maptype in {'rectangle', 'rectanglenavonly'}:
        return obj_names[mat[pos.x][pos.y]]
    elif maptype == 'ring':
        return obj_names[mat[pos]]
    elif maptype == "triangle":
        return obj_names[mat[pos.x][pos.y]]
    elif maptype == "hexagon":
        return obj_names[mat[pos.x][pos.y]]
    else:
        raise NotImplementedError()

def convert2language(obj_names, path, actions, size, maptype, mat):
    """

    Example:
        You are at the bottom left corner of a 2 by 2 grid, 
        where you find an apple. You move up by one step, 
        where you find a computer. You move right by one step, 
        where you find a girl. You move down by one step, 
        where you find a book. You move left by one step. 
        What do you find?
    """
    path_names = []
    pos = path[0]
    obj_name = get_obj_name(mat, pos, obj_names, maptype)
    if isinstance(pos, int):
        path_names.append(int(mat[pos]))
    else:
        path_names.append(int(mat[pos.x][pos.y]))
    txt = init_cond(obj_name, pos, size, maptype, mat)
    for i in range(len(actions)):
        pos = path[i+1]
        obj_name = get_obj_name(mat, pos, obj_names, maptype)
        is_last = i==len(actions) if maptype in {"rectangle", "rectanglenavonly"} else (i==len(actions)-1)
        txt += get_description(actions[i], obj_name, maptype, 
                               is_last=is_last)
        if isinstance(pos, int) or isinstance(pos, np.int64):
            path_names.append(int(mat[pos]))
        else:
            path_names.append(int(mat[pos.x][pos.y]))
    if maptype in {'rectangle', 'rectanglenavonly'}:
        txt += "It is at this point that you realize you have visited all locations on the grid. What is the height and width of the grid? "
        dic = {"question": txt, "answer": f"height={size[1]}, width={size[0]}", "path" :path_names}
    else:
        dic = {"question": txt, "answer": obj_name, "path": path_names}
    return dic


def cot_path_description(act, obj_name, maptype, is_last):
    act_to_string = get_act_to_string(maptype)
    string = f"around the ring by {str(np.abs(act))}"
    string += " step " if np.abs(act) == 1 else " steps "
    string += "in a "+ act_to_string[np.sign(act)] +" direction"
    if is_last:
        string += ", "
    else:
        string += ", "
    return string

def get_cot_with_coordinates_ring(obj_names, path, actions, size, mat):
    """
    Example:
    """
    pos = path[0]
    obj_name = obj_names[mat[pos]]
    txt = f"You can describe your movements in a circular path consisting of {size} connected dots as follows:"
    txt += f" Starting from the {obj_names[mat[pos]]}, you move "
    for i in range(len(actions)):
        pos = path[i+1]
        obj_name = obj_names[mat[pos]]
        txt += cot_path_description(actions[i], obj_name, "ring", 
                               is_last=(i==len(actions)-1))
        txt += f"from the index {path[i]} to the index {pos}, where you will find the {obj_name}. "
        if i < (len(actions) -1):
            txt += "You move "
    txt += "Therefore, the answer is the " + obj_name + "."
    return txt


def get_cot_with_coordinates_square(obj_names, path, actions, size, mat):
    """
    Input:
        dic: dictionary with keys "question" and "answer"

    Example: (Correct answer is "gibbon")

    Prompt:
    You are at the bottom right corner of a 2 by 2 grid, where you find gibbon. 
    You move left by one step, where you find spotlight. You move up by one step, 
    where you find combination lock. You move right by one step, where you find poke bonnet. 
    You move down by one step. What do you find?

    Answer:
    We can describe our movements in the 2 by 2 grid starting from the bottom right corner as follows:
    - Move left from (2,2) to (1,2)
    - Move down from (1,2) to (1,1)
    - Move right from (1,1) to (2,1)
    - Move up from (2,1) to (2,2)
    As a result, we reach the coordinate (2,2) where we find the gibbon.
    Therefore, the answer is gibbon.
    """
    pos = path[0]
    txt = "You can describe your movements in the "+str(size)+" by "+str(size)+" square grid starting from the "+get_init_loc_to_string("square", size)[(pos.x, pos.y)]+" as follows:\n"
    for i in range(len(actions)):
        pos = path[i+1]
        txt += "- Move "+get_act_to_string("square")[(actions[i].x, actions[i].y)]+" from ("+str(path[i].x)+","+str(path[i].y)+") to ("+str(pos.x)+","+str(pos.y)+")"
        txt += f", where you will find the {obj_names[mat[pos.x][pos.y]]}.\n"
        if i == len(actions)-1:
            txt += "As a result, you will reach the coordinate ("+str(pos.x)+","+str(pos.y)+") where you will find the "+obj_names[mat[pos.x][pos.y]]+"."
            txt += " Therefore, the answer is "+obj_names[mat[pos.x][pos.y]]+"."
    return txt

def test_get_cot_with_coordinates_square():
    obj_names = ["gibbon", "spotlight", "combination lock", "poke bonnet"]
    path = [Vec2D(1,1), Vec2D(1,0), Vec2D(0,0), Vec2D(0,1), Vec2D(1,1)]
    actions = [Vec2D(-1,0), Vec2D(0,-1), Vec2D(1,0), Vec2D(0,1)]
    size = 2
    mat = np.array([[0, 2], [1, 3]])
    print(get_cot_with_coordinates_square(obj_names, path, actions, size, mat))


def sample_path(maptype, mat, size):
    """
    Sample a path in the map
    """
    if maptype == "square":
        return sample_path_square(mat, size)
    elif maptype == "rhombus":
        return sample_path_square(mat, size)
    elif maptype in {"rectangle", "rectanglenavonly"}:
        return sample_path_rectangle(mat, size)
    elif maptype == "ring":
        return sample_path_ring(mat, size)
    elif maptype == "hexagon":
        return sample_path_hexagon(mat, size)
    elif maptype == "triangle":
        return sample_path_triangle(mat, size)
    else:
        raise NotImplementedError


def size2str(size):
    if isinstance(size, int):
        return str(size)
    elif isinstance(size, list):
        return "by".join([str(s) for s in reversed(size)])


def main(args):
    with open(args.label_path) as f:
        obj_names = json.load(f) 
    if args.cot_type is not None:
        cot_list = [get_cot_with_map, get_cot_with_coordinates_square,
                    get_cot_with_coordinates_ring]
        cot_func = [f.__name__ for f in cot_list]
        assert args.cot_type in cot_func, "cot_type must be one of {}".format(cot_func)
        cot_dict = dict(zip(cot_func, cot_list))
    dic_list = []
    if args.maptype in {"rectangle", "rectanglenavonly"}:
        from find_all_path import find_paths
        all_rec_paths, all_rec_actions = find_paths(args.size[1], args.size[0])
    while True:
        if len(dic_list) == args.n_sample:
            break
        mat = generate_map(args.maptype, args.size, len(obj_names))
        mat_cp = mat.copy()
        if args.maptype in {'rectangle', 'rectanglenavonly'}:
            path, actions = sample_path_rectangle(all_rec_paths, all_rec_actions)
        else:
            path, actions = sample_path(args.maptype, mat_cp, args.size)
        dic = convert2language(obj_names, path, actions, args.size, args.maptype, mat)
        dic['mat'] = mat.tolist()
        if args.cot_type:
            cot = cot_dict[args.cot_type](obj_names, path, actions, args.size, mat)
            dic.update({"cot": cot})
        dic_list.append(dic)

    if args.cot_type is not None:
        filename = "type-"+args.maptype+"_size-"+size2str(args.size)+"_seed-"+str(args.seed)+\
                "_cot-"+args.cot_type+"_n-"+str(args.n_sample) + "_label-"+args.label_path.split('/')[-1].split('.')[0]+\
                "_sample-"+args.sample_type
    else:
        filename = "type-"+args.maptype+"_size-"+size2str(args.size)+"_seed-"+str(args.seed)+\
                "_n-"+str(args.n_sample) + "_label-"+args.label_path.split('/')[-1].split('.')[0]+\
                "_sample-"+args.sample_type
    if not args.dryrun:
        with open(os.path.join(args.out_dir, filename+'.jsonl'), "w") as outfile:
            for entry in dic_list:
                json.dump(entry, outfile)
                outfile.write('\n')
    
if __name__ == "__main__":
    parser.add_argument('--dryrun', action='store_true', help='dry run')
    parser.add_argument('--seed', type=int, required=True, help='random seed')
    parser.add_argument('--size', type=int, nargs="+", required=True, help='size of structure')
    parser.add_argument('--n_sample', type=int, required=True, help='number of questions to generate')
    parser.add_argument('--maptype', type=str, required=True, help='The structure type')
    parser.add_argument('--label_path', type=str, required=True, help='The txt file for object names')
    parser.add_argument('--out_dir', type=str, required=False, default="./", help='default folder to save the generated questions')
    parser.add_argument('--cot_type', type=str, default=None, help='The cot type, one of\
                        [get_cot_with_map, get_cot_with_coordinates]')
    parser.add_argument('--sample_type', type=str, default="greedy", help='The sampling type, one of [greedy]')
    # Parse arguments
    args = parser.parse_args()
    if len(args.size) == 1:
        args.size = args.size[0]
    np.random.seed(args.seed) 
    main(args)


