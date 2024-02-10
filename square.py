from sample_maps import *
import itertools
import random

article = inflect.engine()

# ref: https://stackoverflow.com/questions/9647202/ordinal-numbers-replacement
def make_ordinal(n):
    '''
    Convert an integer into its ordinal representation::

        make_ordinal(0)   => '0th'
        make_ordinal(3)   => '3rd'
        make_ordinal(122) => '122nd'
        make_ordinal(213) => '213th'
    '''
    n = int(n)
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    return str(n) + suffix


def print_grid(mat, size, obj_names):
    string = f"You have been given a {size} by {size} square grid. "
    temporal_order = []
    for i in reversed(range(size)):
        string += f"In the {make_ordinal(size-i)} row, from left to right, we have "
        for j in range(size - 1):
            obj_idx = int(mat[j][i])
            obj_name = obj_names[obj_idx]
            temporal_order.append(obj_idx)
            if size == 2:
                string += f"{article.a(obj_name)} "
            else:
                string += f"{article.a(obj_name)}, "
        obj_idx = int(mat[size-1][i])
        obj_name = obj_names[obj_idx]
        temporal_order.append(obj_idx)
        string += f"and {article.a(obj_name)}. "

    assert (len(temporal_order) == size**2)

    return string, temporal_order

def random_grid(mat, size, obj_names):
    string = f"You have been given a {size} by {size} square grid with various items located at different indices: "
    grid_coords = list(itertools.product(range(1, size+1), range(1, size+1)))
    random.shuffle(grid_coords)
    temporal_order = []
    for i in range(len(grid_coords)):
        coord = grid_coords[i]
        obj_idx = int(mat[coord[0]-1][coord[1]-1])
        obj_name = obj_names[obj_idx]
        temporal_order.append(obj_idx)
        if i == len(grid_coords)-1:
            string += f"and {article.a(obj_name)} is at index {coord}. "
        else:
            string += f"{article.a(obj_name)} is at index {coord}, "

    assert (len(temporal_order) == size**2)

    return string, temporal_order

def snake_grid(mat, size, obj_names, with_coord):
    string = f"You have been given a {size} by {size} square grid. "
    string += f"Starting from a vertex, you will move along the edges of the grid. "
    string += f"Initially, you are positioned at the bottom-left corner of the grid, "
    temporal_order = []
    for i in range(size):
        if i%2 == 0:                
            for j in range(size - 1):
                obj_idx = int(mat[j][i])
                obj_name = obj_names[obj_idx]
                temporal_order.append(obj_idx)
                if with_coord:
                    string += f"where you will find {article.a(obj_name)} at index ({j+1},{i+1}), then you go right, "
                else:
                    string += f"where you will find {article.a(obj_name)}, then you go right, "
            obj_idx = int(mat[size-1][i])
            obj_name = obj_names[obj_idx]
            temporal_order.append(obj_idx)
            if with_coord:
                string += f"where you will find {article.a(obj_name)} at index ({size},{i+1}). "
            else:
                string += f"where you will find {article.a(obj_name)}. "
            if i < size - 1:
                string += f"Then you go up, "
        elif i%2 == 1:
            for j in reversed(range(1,size)):
                obj_idx = int(mat[j][i])
                obj_name = obj_names[obj_idx]
                temporal_order.append(obj_idx)
                if with_coord:
                    string += f"where you will find {article.a(obj_name)} at index ({j+1},{i+1}), then you go left, "
                else:
                    string += f"where you will find {article.a(obj_name)}, then you go left, "
            obj_idx = int(mat[0][i])
            obj_name = obj_names[obj_idx]
            temporal_order.append(obj_idx)
            if with_coord:
                string += f"where you will find {article.a(obj_name)} at index (1,{i+1}). "
            else:
                string += f"where you will find {article.a(obj_name)}. "
            if i < size - 1:
                string += f"Then you go up, "
    string += "Now you have all the information on the map. "
    
    assert (len(temporal_order) == size**2)
    return string, temporal_order


def sample_neighbor(mat, size, steps):
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


    start = Vec2D(np.random.choice(range(size-1),1)[0], np.random.choice(range(size-1),1)[0]) # start at a random location in the mat
    commands = [Vec2D(0, 1), Vec2D(1, 0), Vec2D(0, -1), Vec2D(-1, 0)]
    moves = ['up', 'right', 'down', 'left']
    p_dist = [0.25, 0.25, 0.25, 0.25]
    prev = start
    path = [start]
    mat_cp = mat.copy()
    names = [mat[start.x, start.y]]
    mat[start.x, start.y] = -1
    actions = []
    directions = []
    step = 0

    while step < steps:
        ind =  np.random.choice(range(4), p=p_dist)
        command = commands[ind]
        move = moves[ind]
        pos = prev + command 
        if is_valid(pos):
            path.append(pos)
            actions.append(command)
            directions.append(move)
            names.append(mat_cp[pos.x, pos.y])
            p_dist = get_pdist(command)
            if visited(pos):
                return names, directions, path, actions
            else:
                mat[pos.x, pos.y] = -1
                prev = pos
                step += 1
    return names, directions, path, actions

def path2language(names, directions, obj_names):
    string = f"You start at the position where the {obj_names[names[0]]} is located, "
    if len(directions) == 1:
        string += f"then you go {directions[0]} by one step. "
    else:
        for i in range(len(directions)-1):
            string += f"then you go {directions[i]} by one step, "
        string += f"and then you go {directions[-1]} by one step. "
    string += "What will you find?"
    print(directions, names)
    answer = f"{obj_names[names[-1]]}."
    return string, answer


def convert2qapairs(mat, size, names, directions, obj_names, special_order):
    if special_order == "snake_order":
         txt, temporal_order = snake_grid(mat, size, obj_names)
    elif special_order == "snake_order_with_coordinates":
         txt = snake_grid(mat, size, obj_names, True)
    elif special_order == "random_order":
         txt, temporal_order = random_grid(mat, size, obj_names)
    else:
        txt, temporal_order = print_grid(mat, size, obj_names)
    string, answer = path2language(names, directions, obj_names) 
    question = txt + string
    dic = {"question": question, "answer": answer, 
        'struct': mat.tolist(), 'temporal_order': temporal_order}    
    return dic

def cot2language(mat, size, obj_names, path, actions):
    pos = path[0]
    txt = f"You can describe your movements in the {size} by {size} square grid as follows: " 
    txt += f"Starting at the {obj_names[mat[pos.x][pos.y]]}, "
    for i in range(len(actions)):
        pos = path[i+1]
        txt += "- Move "+get_act_to_string("square")[(actions[i].x, actions[i].y)]+" from ("+str(path[i].x)+","+str(path[i].y)+") to ("+str(pos.x)+","+str(pos.y)+")"
        txt += f", where you will find the {obj_names[mat[pos.x][pos.y]]}.\n"
        if i == len(actions)-1:
            txt += "As a result, you will reach the coordinate ("+str(pos.x)+","+str(pos.y)+") where you will find the "+obj_names[mat[pos.x][pos.y]]+"."
            txt += " Therefore, the answer is "+obj_names[mat[pos.x][pos.y]]+"."
    return txt

def main(args):
    with open(args.label_path) as f:
        obj_names = json.load(f)
    dic_list = []
    for i in range(args.n_sample):
        print(f"iteration {i}")
        mat = generate_map(args.maptype, args.size, len(obj_names))
        mat_cp = mat.copy()
        names, directions, path, actions = sample_neighbor(mat_cp, args.size, args.steps)
        if args.cot_type == "get_cot_with_coordinates":
            mat_cp_dic = mat.copy()
            dic = convert2qapairs(mat_cp_dic, args.size, names, directions, obj_names, args.special_order)
            cot_txt = cot2language(mat, args.size, obj_names, path, actions)
            dic.update({"cot": cot_txt})
        else:
            assert args.cot_type == None
            dic = convert2qapairs(mat, args.size, names, directions, obj_names, args.special_order)
        dic_list.append(dic)

    if args.cot_type == "get_cot_with_coordinates":
        filename = "type-"+args.maptype+"_size-"+str(args.size)+"_steps-"+str(args.steps)+"_seed-"+str(args.seed)+\
                "_cot-"+args.cot_type+"_n-"+str(args.n_sample)  + "_special_order-"+str(args.special_order)
    else:
        assert args.cot_type == None
        filename = "type-"+args.maptype+"_size-"+str(args.size)+"_steps-"+str(args.steps)+"_seed-"+str(args.seed)+\
                "_n-"+str(args.n_sample) + "_special_order-"+str(args.special_order)
    with open(os.path.join(args.out_dir, filename+'.jsonl'), "w") as outfile:
        for entry in dic_list:
            json.dump(entry, outfile)
            outfile.write('\n')

if __name__ == "__main__":
    parser.add_argument('--seed', type=int, required=True, help='random seed')
    parser.add_argument('--size', type=int, required=True, help='size of structure')
    parser.add_argument('--steps', type=int, required=True, help='the number of steps from the node')
    parser.add_argument('--n_sample', type=int, required=True, help='number of questions to generate')
    parser.add_argument('--maptype', type=str, required=True, help='The structure type')
    parser.add_argument('--label_path', type=str, required=True, help='The txt file for object names')
    parser.add_argument('--out_dir', type=str, required=False, default="./", help='default folder to save the generated questions')
    parser.add_argument('--cot_type', type=str, default=None, help='The cot type, one of\
                        [get_cot_with_map, get_cot_with_coordinates]')
    parser.add_argument('--special_order', type=str, required=False, default = False, help='Describe the full structure in different order, \
                        default is the original order, if you want to specify, one of [snake_order, random_order]')
    
    # Parse arguments
    args = parser.parse_args()
    np.random.seed(args.seed) 
    main(args)

# run using python /square.py --seed 4 --size 4  --steps 4 --maptype square --cot_type get_cot_with_coordinates  --label_path imagenet-simple-labels.json --n_sample 100 --out_dir ./memory_global
