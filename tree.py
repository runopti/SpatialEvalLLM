import networkx as nx
import numpy as np
import json
import argparse
import os
import inflect
import random
import pickle

article = inflect.engine()

parser = argparse.ArgumentParser(description='Description of your script.')

def generate_tree(maptype, size, obj_names):
    if maptype == 'tree':
        arr = np.random.choice(obj_names, size=size, replace=False)
        tree = nx.random_tree(n=size, create_using=nx.DiGraph)  
        tree = nx.relabel_nodes(tree, lambda x: arr[x], copy=True) # relabeled tree
    else:
        raise NotImplementedError()
    return tree

def get_root(tree):
    for node, in_degree in tree.in_degree():
        if in_degree==0:
            return node
    print(tree.in_degree())
    raise ValueError("could not find root")

def print_tree_from_node(tree, node):
    """
    Example:
    'The root is slug. slug has children toy terrier and scorpion. 
    toy terrier has children limousine and binoculars and tank suit. 
    limousine has no children. 
    """

    children = list(tree.successors(node))
    if len(children) == 0:
        string = f"The {node} has no children."
        return string
    elif len(children) == 1:
        string = f"The {node} has a child: {article.a(children[0])}. " 
        children_strings = [print_tree_from_node(tree, child) for child in children]
        string += ' '.join(children_strings)
        return string
    else:
        num_of_children = len(children)
        string = f"The {node} has {num_of_children} children: "
        for i in range(len(children)):
            if i == len(children)-1:
                if len(children) != 2:
                    string += f", and {article.a(children[i])}. "
                else:
                    string += f" and {article.a(children[i])}. "
            else:
                if i == 0:
                    string += f"{article.a(children[i])}"        
                else:
                    string += f", {article.a(children[i])}"        
            
        children_strings = [print_tree_from_node(tree, child) for child in children]
        string += ' '.join(children_strings)
        return string

def print_tree_from_node_breadth_first(tree, root):
    """
    Example:
    'The root is slug. slug has children toy terrier and scorpion. 
    toy terrier has children limousine and binoculars and tank suit. 
    limousine has no children. 
    """
    nodes = list(nx.bfs_tree(tree, root).nodes())
    string = f"The root node is {article.a(root)}. "
    for node in nodes:
        children = list(tree.successors(node))
        if len(children) == 0:
            string += f"The {node} has no children. "
        elif len(children) == 1:
            string += f"The {node} has a child: {article.a(children[0])}. " 
        else:
            num_of_children = len(children)
            string += f"The {node} has {num_of_children} children: "
            for i in range(len(children)):
                if i == len(children)-1:
                    if len(children) != 2:
                        string += f", and {article.a(children[i])}. "
                    else:
                        string += f" and {article.a(children[i])}. "
                else:
                    if i == 0:
                        string += f"{article.a(children[i])}"        
                    else:
                        string += f", {article.a(children[i])}"        
                
    return string    

def print_tree(tree):
    root = get_root(tree)
    string = f"The root node is {article.a(root)}. "
    string += print_tree_from_node(tree, root)
    return string

def print_tree_breadth_first(tree):
    root = get_root(tree)
    string = print_tree_from_node_breadth_first(tree, root)
    return string

# helper function of finding relatives
def find_siblings(tree, node):
    root = get_root(tree)
    if node != root:
        parent = list(tree.predecessors(node))[0] # tree so have only one parent
        siblings = [sibling for sibling in tree.successors(parent) if sibling!=node]
        return siblings
    else:
        return ""

def find_parent(tree, node):
    root = get_root(tree)
    if node != root:
        parent = list(tree.predecessors(node))
        return parent
    else:
        return ""

def find_greatgreatgrandparent(tree, node):
    grandparent = find_grandparent(tree, node)
    if grandparent == "":
        return ""
    greatgreatgrandparent = find_grandparent(tree, grandparent[0])
    return greatgreatgrandparent


def find_grandparent(tree, node):
    root = get_root(tree)
    if (node == root) or (node in tree.successors(root)):
        return ''
    else:
        parent = list(tree.predecessors(node))[0] 
        grandparent = list(tree.predecessors(parent))
        return grandparent

def find_greatgreatgrandchildren(tree, node):
    grandchildren = find_grandchildren(tree, node)
    grandgrandchildren = []
    for grandchild in grandchildren:
        tmp = find_grandchildren(tree, grandchild)
        if len(tmp) > 0 and isinstance(tmp, list):
            grandgrandchildren += tmp
    if len(grandgrandchildren) == 0:
        return ''
    return grandgrandchildren


def find_grandchildren(tree, node):
    children = tree.successors(node)
    grandchildren = []
    for child in children: 
        for grandchild in tree.successors(child):
            grandchildren.append(grandchild)
    return grandchildren

def find_uncles(tree, node):
    root = get_root(tree)
    if node == root:
        return ""
    else:
        parent = list(tree.predecessors(node))[0] 
        parent_siblings = find_siblings(tree, parent)
        return parent_siblings

def find_cousins(tree, node):
    root = get_root(tree)
    if node == root:
        return ""
    else:
        parent = list(tree.predecessors(node))[0] 
        uncles = find_uncles(tree, node)
        if uncles == "":
            return ""
        else:
            cousins = []
            for uncle in uncles:
                if uncle != parent:
                    cousins += [cousin for cousin in tree.successors(uncle)]
            return cousins

# helper function for find common ancestors
def find_ancestors(tree, node):
    ancestors = []
    curr_node = node
    while True:
        if len(list(tree.predecessors(curr_node))) > 0:
            parent = list(tree.predecessors(curr_node))[0]
            ancestors.append(parent)
            curr_node = parent
        else:
            break
    return ancestors

def find_first_common_ancestor(tree, node1, node2):
    node1_ancestors = find_ancestors(tree, node1)
    node2_ancestors = find_ancestors(tree, node2)

    for ancestor in node1_ancestors:
        if ancestor in node2_ancestors:
            return ancestor
    return 'no common ancestor'

def find_depth(tree, node):
    return len(find_ancestors(tree, node))

# helper function for asking questions about relations
def question_parent(tree, node):
    question = f" What is the parent of the {node}? " # 1 step
    #answer = list(tree.predecessors(node))[0]
    answer = find_parent(tree, node)
    return [question, answer]

def question_sibling(tree, node):
    #if node != root:
    question = f" What is the sibling of the {node}? " # 2 steps
    answer = find_siblings(tree, node)
    return [question, answer]

def question_cousin(tree, node):
    question = f" What is the cousin of the {node}? " # 4 steps
    answer = find_cousins(tree, node)
    return [question, answer]

def question_greatgreatgrandparent(tree, node):
    question = f" What is the great-great-grandparent of the {node}? " # 4 steps
    answer = find_greatgreatgrandparent(tree, node)
    return [question, answer]

def question_grandparent(tree, node):
    question = f" What is the grandparent of the {node}? " # 2 steps
    answer = find_grandparent(tree, node)
    return [question, answer]

def question_greatgreatgrandchildren(tree, node):
    question = f" What is(are) the great-great-grandchild(ren) of the {node}? " # 4 steps
    answer = find_greatgreatgrandchildren(tree, node)
    return [question, answer]

def question_grandchildren(tree, node):
    question = f" What is(are) the grandchild(ren) of the {node}? " # 2 steps
    answer = find_grandchildren(tree, node)
    return [question, answer]

def question_uncle(tree, node):
    question = f" What is(are) the uncle(s) of the {node}? " # 3 steps
    answer = find_uncles(tree, node)
    return [question, answer]

def question_random(questions_pairs, tree, node):
    ind = np.random.choice(len(questions_pairs))
    question_pair = questions_pairs[ind] 
    return question_pair(tree, node)



def convert2language(tree, nsteps, traverse_type):
    size = tree.number_of_nodes()
    txt = "You have been given a tree structure with " + str(size) + " nodes. "
    if traverse_type == "BFS":
        txt += print_tree_breadth_first(tree)
    else:
        txt += print_tree(tree)
    if nsteps == 4:
        question_pairs = [question_greatgreatgrandparent, question_greatgreatgrandchildren, question_cousin]
    elif nsteps == None:
        question_pairs = [question_parent, question_sibling, question_cousin, question_uncle, question_grandparent, question_grandchildren]
    else:
        raise NotImplementedError(f"nsteps has to be 4 or None")
    answer = ""
    trial = 0
    while answer == "":
        node = np.random.choice(tree.nodes, size=1, replace=False)[0]
        question_answer = question_random(question_pairs, tree, node)
        txt_part = question_answer[0]
        try:
            answer = ', '.join(question_answer[1])
        except TypeError:
            import ipdb; ipdb.set_trace()
            answer = question_answer[1]
        trial += 1
        if trial == 10:
            return None, False
    txt += txt_part
    dic = {"question": txt, "answer": answer}
    return dic, True
    

def main(args):
    with open(args.label_path) as f:
        obj_names = json.load(f)
    dic_list = []
    obj_list = []
    for i in range(args.n_sample):
        status = False 
        while status == False:
            print(f"iteration {i}")
            tree = generate_tree(args.maptype, args.size, obj_names)
            if args.save_tree:
                out_dir = args.out_dir + "/"+args.maptype+"_size-"+str(args.size)+"_seed-"+str(args.seed)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                with open(os.path.join(out_dir, "id-"+str(i).zfill(3)+".pickle"), "wb") as f:
                    pickle.dump(tree, f)
            dic, status = convert2language(tree, args.steps, args.traverse_type)
        dic_list.append(dic)
        obj_list.append(tree)

    filename = "type-"+args.maptype+"_size-"+str(args.size)+"_steps-"+str(args.steps)+"_seed-"+str(args.seed)+\
            "_n-"+str(args.n_sample)+"_traverse_type-"+str(args.traverse_type)
    with open(os.path.join(args.out_dir, filename+'.jsonl'), "w") as outfile:
        for entry in dic_list:
            json.dump(entry, outfile)
            outfile.write('\n')
    
    # save list of tree objects
    with open(os.path.join(args.out_dir, filename + ".pickle"), "wb") as outfile:
        pickle.dump(obj_list, outfile)
    
if __name__ == "__main__":
    parser.add_argument('--seed', type=int, required=True, help='random seed')
    parser.add_argument('--size', type=int, required=True, help='size of structure')
    parser.add_argument('--steps', type=int, required=True, help='the number of steps from the node')
    parser.add_argument('--n_sample', type=int, required=True, help='number of questions to generate')
    parser.add_argument('--maptype', type=str, required=True, help='The structure type')
    parser.add_argument('--label_path', type=str, required=True, help='The txt file for object names')
    parser.add_argument('--out_dir', type=str, required=False, default="./", help='default folder to save the generated questions')
    parser.add_argument('--save_tree', action='store_true', help='save the generated tree')
    parser.add_argument('--traverse_type', type=str, required=False, default="DFS", help='tree traverse type')

    # Parse arguments
    args = parser.parse_args()
    np.random.seed(args.seed) 
    random.seed(args.seed)
    main(args)

    
