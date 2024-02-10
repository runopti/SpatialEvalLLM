from sample_maps import *
article = inflect.engine()

def print_ring(mat, size, obj_names):
    string = f"You have been given a circular path consisting of {size} connected dots. At the start, you are positioned on the dot that is located at the top of the path, where you find {article.a(obj_names[mat[0]])}. Moving in a clockwise direction from the {obj_names[mat[0]]}, the elements on the path are "
    for i in range(1,size-1):
        obj_name = obj_names[mat[i]]
        string += f"{article.a(obj_name)}, "
    string += f"and {article.a(obj_names[mat[size-1]])}. "
    return string

def sample_path_ring_random(mat, size, steps):
    # Sample a sequence of actions that traverses the ring.
    # The sequence of actions is a list of [clockwise or counter-clockwise, how many steps to take]. 
    start = np.random.choice(size)
    commands = [i for i in range(1, size+1)] + [-1*i for i in range(1, size+1)]
    p_dist = [1.0/(2*size) for i in range(2*size)]
    prev = start
    path = [start]
    mat[start] = -1
    actions = []
    step = 0
    while step < steps:
        command = np.random.choice(commands, p=p_dist)
        pos = (prev + command) % size
        path.append(pos)
        actions.append(command)
        if mat[pos] == -1:
            return path, actions
        else:
            mat[pos] = -1
            prev = pos
        step += 1
    return path, actions

def ring_path_description(act, obj_name, maptype, is_last):
    string = " you move "
    act_to_string = get_act_to_string(maptype)
    string += "around the ring by " + str(np.abs(act)) 
    string += " step " if np.abs(act) == 1 else " steps "
    string += "in a "+ act_to_string[np.sign(act)] +" direction"
    if is_last:
        string += ". What will you find?"
    else:
        string += ", and"
    return string

def ring2language(obj_names, path, actions, size, maptype, mat):
    """

    Example:
        You are at the bottom left corner of a 2 by 2 grid, 
        where you find an apple. You move up by one step, 
        where you find a computer. You move right by one step, 
        where you find a girl. You move down by one step, 
        where you find a book. You move left by one step. 
        What do you find?
    """
    pos = path[0]
    obj_name = obj_names[mat[pos]]
    txt = print_ring(mat, size, obj_names)
    txt += f"Starting from the {obj_names[mat[pos]]},"
    for i in range(len(actions)):
        pos = path[i+1]
        obj_name = obj_names[mat[pos]]
        txt += ring_path_description(actions[i], obj_name, maptype, 
                               is_last=(i==len(actions)-1))
    dic = {"question": txt, "answer": obj_name}
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

def cot2language(obj_names, path, actions, size, maptype, mat):
    """

    Example:
        You are at the bottom left corner of a 2 by 2 grid, 
        where you find an apple. You move up by one step, 
        where you find a computer. You move right by one step, 
        where you find a girl. You move down by one step, 
        where you find a book. You move left by one step. 
        What do you find?
    """
    pos = path[0]
    obj_name = obj_names[mat[pos]]
    description = print_ring(mat, size, obj_names)
    txt = f"You can describe your movements in a circular path consisting of {size} connected dots as follows:"
    txt += f" Starting from the {obj_names[mat[pos]]}, you move "
    for i in range(len(actions)):
        pos = path[i+1]
        obj_name = obj_names[mat[pos]]
        txt += cot_path_description(actions[i], obj_name, maptype, 
                               is_last=(i==len(actions)-1))
        txt += f"from the index {path[i]} to the index {pos}, where you will find the {obj_name}. "
        if i < (len(actions) -1):
            txt += "You move "
    txt += "Therefore, the answer is the " + obj_name + "."
    return txt


def main(args):
    with open(args.label_path) as f:
        obj_names = json.load(f)
    dic_list = []
    for i in range(args.n_sample):
        print(f"iteration {i}")
        mat = generate_map(args.maptype, args.size, len(obj_names))
        mat_cp = mat.copy()
        path, actions = sample_path_ring_random(mat_cp, args.size, args.steps)
        if args.cot_type == "get_cot_with_coordinates":
            dic = ring2language(obj_names, path, actions, args.size, args.maptype, mat)
            cot_txt = cot2language(obj_names, path, actions, args.size, args.maptype, mat)
            dic.update({"cot": cot_txt})
        else:
            assert args.cot_type == None
            dic = ring2language(obj_names, path, actions, args.size, args.maptype, mat)
        dic_list.append(dic)

    if args.cot_type == "get_cot_with_coordinates":
        filename = "type-"+args.maptype+"_size-"+str(args.size)+"_steps-"+str(args.steps)+"_seed-"+str(args.seed)+\
                "_cot-"+args.cot_type+"_n-"+str(args.n_sample)
    else:
        assert args.cot_type == None
        filename = "type-"+args.maptype+"_size-"+str(args.size)+"_steps-"+str(args.steps)+"_seed-"+str(args.seed)+\
                "_n-"+str(args.n_sample)
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
    # Parse arguments
    args = parser.parse_args()
    np.random.seed(args.seed) 
    assert "memory_global" in args.out_dir
    main(args)

# run ring.py --seed 2 --size 10  --steps 3 --maptype ring --label-path imagenet-simple-labels.json --n-sample 100
# {"question": "You have been given a circular path consisting of 6 connected dots. 
# At the start, you are positioned on the dot that is located at the top of the path, where you find running shoe.
# Moving in a clockwise direction from the running shoe, the elements on the path are sombrero, great grey owl, Welsh Springer Spaniel, patas monkey, and handkerchief. 
# Starting from running shoe, you move around the ring by 5 steps in a clockwise direction, 
# and you move around the ring by 1 steps in a counter-clockwise direction, 
# and you move around the ring by 1 steps in a counter-clockwise direction. 
# What do you find?", "answer": "Welsh Springer Spaniel"}



# "cot": "We can describe our movements in a circular path consisting of 6 connected dots as follows:
# Starting from running shoe, you move around the ring by 5 steps in a clockwise direction, where you find handkerchief. 
# Starting from handkerchief, you move around the ring by 1 steps in a counter-clockwise direction, where you find patas monkey.
# Starting from patas monkey, you move around the ring by 1 steps in a counter-clockwise direction, where you find Welsh Springer Spaniel.
 
