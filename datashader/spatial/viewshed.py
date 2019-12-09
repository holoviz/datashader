from math import atan, sqrt, fabs
from math import pi as PI

import numpy as np
import numba as nb
from numba import jit

import xarray

E_ROW_ID = 0
E_COL_ID = 1
E_TYPE_ID = 2

E_ANG_ID = 3
E_ELEV_0 = 4
E_ELEV_1 = 5
E_ELEV_2 = 6

AE_ANG_ID = 0
AE_ELEV_0 = 1
AE_ELEV_1 = 2
AE_ELEV_2 = 3

TN_KEY_ID = 0
TN_GRAD_0 = 1
TN_GRAD_1 = 2
TN_GRAD_2 = 3
TN_ANG_0 = 4
TN_ANG_1 = 5
TN_ANG_2 = 6
TN_MAX_GRAD_ID = 7

TN_COLOR_ID = 0
TN_LEFT_ID = 1
TN_RIGHT_ID = 2
TN_PARENT_ID = 3

NIL_ID = -1

# view options default values
OBS_ELEV = 0
TARGET_ELEV = 0

# if a cell is invisible, its value is set to -1
INVISIBLE = -1

# color of node in red-black Tree
RB_RED = 0
RB_BLACK = 1

# event type
ENTERING_EVENT = 1
EXITING_EVENT = -1
CENTER_EVENT = 0

# this value is returned by findMaxValueWithinDist() if there is no key within
# that distance
SMALLEST_GRAD = -9999999999999999999999.0


@jit(nb.i8(nb.f8, nb.f8), nopython=True)
def _compare(a, b):
    if a < b:
        return -1
    if a > b:
        return 1
    return 0


@jit(nb.f8(nb.f8[:, :], nb.i8), nopython=True)
def _find_value_min_value(tree_vals, node_id):
    return min(tree_vals[node_id][TN_GRAD_0],
               tree_vals[node_id][TN_GRAD_1],
               tree_vals[node_id][TN_GRAD_2])


def _print_tree(status_struct):
    for i in range(len(status_struct)):
        print(i, status_struct[i][0])


def _print_tv(tv):
    print('key=', tv[TN_KEY_ID],
          'grad=', tv[TN_GRAD_0], tv[TN_GRAD_1], tv[TN_GRAD_2],
          'ang=', tv[TN_ANG_0], tv[TN_ANG_1], tv[TN_ANG_2],
          'max_grad=', tv[TN_MAX_GRAD_ID])
    return


@jit(nb.void(nb.f8[:, :], nb.i8[:, :], nb.i8, nb.f8[:], nb.i8), nopython=True)
def _create_tree_nodes(tree_vals, tree_nodes, x, val, color=RB_RED):
    # Create a TreeNode using given TreeValue

    # every node has null nodes as children initially, create one such object
    # for easy management

    tree_vals[x][TN_KEY_ID] = val[TN_KEY_ID]
    tree_vals[x][TN_GRAD_0] = val[TN_GRAD_0]
    tree_vals[x][TN_GRAD_1] = val[TN_GRAD_1]
    tree_vals[x][TN_GRAD_2] = val[TN_GRAD_2]
    tree_vals[x][TN_ANG_0] = val[TN_ANG_0]
    tree_vals[x][TN_ANG_1] = val[TN_ANG_1]
    tree_vals[x][TN_ANG_2] = val[TN_ANG_2]
    tree_vals[x][TN_MAX_GRAD_ID] = SMALLEST_GRAD

    tree_nodes[x][TN_COLOR_ID] = color
    tree_nodes[x][TN_LEFT_ID] = NIL_ID
    tree_nodes[x][TN_RIGHT_ID] = NIL_ID
    tree_nodes[x][TN_PARENT_ID] = NIL_ID
    return


@jit(nb.i8(nb.i8[:, :], nb.i8), nopython=True)
def _tree_minimum(tree_nodes, x):
    while tree_nodes[x][TN_LEFT_ID] != NIL_ID:
        x = tree_nodes[x][TN_LEFT_ID]
    return x


# function used by deletion
@jit(nb.i8(nb.i8[:, :], nb.i8), nopython=True)
def _tree_successor(tree_nodes, x):
    # Find the highest successor of a node in the tree

    if tree_nodes[x][TN_RIGHT_ID] != NIL_ID:
        return _tree_minimum(tree_nodes, tree_nodes[x][TN_RIGHT_ID])

    y = tree_nodes[x][TN_PARENT_ID]
    while y != NIL_ID and x == tree_nodes[y][TN_RIGHT_ID]:
        x = y
        if tree_nodes[y][TN_PARENT_ID] == NIL_ID:
            return y
        y = tree_nodes[y][TN_PARENT_ID]
    return y


@jit(nb.f8(nb.f8[:]), nopython=True)
def _find_max_value(node_value):
    # Find the max value in the given tree.
    return node_value[TN_MAX_GRAD_ID]


@jit(nb.i8(nb.f8[:, :], nb.i8[:, :], nb.i8, nb.i8), nopython=True)
def _left_rotate(tree_vals, tree_nodes, root, x):
    # A utility function to left rotate subtree rooted with a node.

    y = tree_nodes[x][TN_RIGHT_ID]

    # fix x
    x_left = tree_nodes[x][TN_LEFT_ID]
    y_left = tree_nodes[y][TN_LEFT_ID]
    if tree_vals[x_left][TN_MAX_GRAD_ID] > tree_vals[y_left][TN_MAX_GRAD_ID]:
        tmp_max = tree_vals[x_left][TN_MAX_GRAD_ID]
    else:
        tmp_max = tree_vals[y_left][TN_MAX_GRAD_ID]

    min_value = _find_value_min_value(tree_vals, x)
    if tmp_max > min_value:
        tree_vals[x][TN_MAX_GRAD_ID] = tmp_max
    else:
        tree_vals[x][TN_MAX_GRAD_ID] = min_value

    # fix y
    y_right = tree_nodes[y][TN_RIGHT_ID]
    if tree_vals[x][TN_MAX_GRAD_ID] > tree_vals[y_right][TN_MAX_GRAD_ID]:
        tmp_max = tree_vals[x][TN_MAX_GRAD_ID]
    else:
        tmp_max = tree_vals[y_right][TN_MAX_GRAD_ID]

    min_value = _find_value_min_value(tree_vals, y)
    if tmp_max > min_value:
        tree_vals[y][TN_MAX_GRAD_ID] = tmp_max
    else:
        tree_vals[y][TN_MAX_GRAD_ID] = min_value

    # left rotation
    # see pseudo code on page 278 CLRS

    # turn y's left subtree into x's right subtree
    tree_nodes[x][TN_RIGHT_ID] = tree_nodes[y][TN_LEFT_ID]
    y_left = tree_nodes[y][TN_LEFT_ID]
    tree_nodes[y_left][TN_PARENT_ID] = x
    # link x's parent to y
    tree_nodes[y][TN_PARENT_ID] = tree_nodes[x][TN_PARENT_ID]

    if tree_nodes[x][TN_PARENT_ID] == NIL_ID:
        root = y
    else:
        x_parent = tree_nodes[x][TN_PARENT_ID]
        if x == tree_nodes[x_parent][TN_LEFT_ID]:
            tree_nodes[x_parent][TN_LEFT_ID] = y
        else:
            tree_nodes[x_parent][TN_RIGHT_ID] = y

    tree_nodes[y][TN_LEFT_ID] = x
    tree_nodes[x][TN_PARENT_ID] = y
    return root


@jit(nb.i8(nb.f8[:, :], nb.i8[:, :], nb.i8, nb.i8), nopython=True)
def _right_rotate(tree_vals, tree_nodes, root, y):
    # A utility function to right rotate subtree rooted with a node.

    x = tree_nodes[y][TN_LEFT_ID]

    # fix y
    x_right = tree_nodes[x][TN_RIGHT_ID]
    y_right = tree_nodes[y][TN_RIGHT_ID]
    if tree_vals[x_right][TN_MAX_GRAD_ID] > tree_vals[y_right][TN_MAX_GRAD_ID]:
        tmp_max = tree_vals[x_right][TN_MAX_GRAD_ID]
    else:
        tmp_max = tree_vals[y_right][TN_MAX_GRAD_ID]

    min_value = _find_value_min_value(tree_vals, y)
    if tmp_max > min_value:
        tree_vals[y][TN_MAX_GRAD_ID] = tmp_max
    else:
        tree_vals[y][TN_MAX_GRAD_ID] = min_value

    # fix x
    x_left = tree_nodes[x][TN_LEFT_ID]
    if tree_vals[x_left][TN_MAX_GRAD_ID] > tree_vals[y][TN_MAX_GRAD_ID]:
        tmp_max = tree_vals[x_left][TN_MAX_GRAD_ID]
    else:
        tmp_max = tree_vals[y][TN_MAX_GRAD_ID]

    min_value = _find_value_min_value(tree_vals, x)
    if tmp_max > min_value:
        tree_vals[x][TN_MAX_GRAD_ID] = tmp_max
    else:
        tree_vals[x][TN_MAX_GRAD_ID] = min_value

    # rotation
    tree_nodes[y][TN_LEFT_ID] = tree_nodes[x][TN_RIGHT_ID]
    x_right = tree_nodes[x][TN_RIGHT_ID]
    tree_nodes[x_right][TN_PARENT_ID] = y

    tree_nodes[x][TN_PARENT_ID] = tree_nodes[y][TN_PARENT_ID]

    if tree_nodes[y][TN_PARENT_ID] == NIL_ID:
        root = x
    else:
        y_parent = tree_nodes[y][TN_PARENT_ID]
        if tree_nodes[y_parent][TN_LEFT_ID] == y:
            tree_nodes[y_parent][TN_LEFT_ID] = x
        else:
            tree_nodes[y_parent][TN_RIGHT_ID] = x

    tree_nodes[x][TN_RIGHT_ID] = y
    tree_nodes[y][TN_PARENT_ID] = x
    return root


@jit(nb.i8(nb.f8[:, :], nb.i8[:, :], nb.i8, nb.i8), nopython=True)
def _rb_insert_fixup(tree_vals, tree_nodes, root, z):
    # Fix red-black tree after insertion. This may change the root pointer.

    # see pseudocode on page 281 in CLRS
    z_parent = tree_nodes[z][TN_PARENT_ID]
    while tree_nodes[z_parent][TN_COLOR_ID] == RB_RED:
        z_parent_parent = tree_nodes[z_parent][TN_PARENT_ID]
        n1 = tree_nodes[z][TN_PARENT_ID]
        n2 = tree_nodes[z_parent_parent][TN_LEFT_ID]
        if n1 == n2:
            y = tree_nodes[z_parent_parent][TN_RIGHT_ID]
            if tree_nodes[y][TN_COLOR_ID] == RB_RED:
                # case 1
                tree_nodes[z_parent][TN_COLOR_ID] = RB_BLACK
                tree_nodes[y][TN_COLOR_ID] = RB_BLACK
                tree_nodes[z_parent_parent][TN_COLOR_ID] = RB_RED
                # re assignment for z
                z = z_parent_parent
            else:
                if z == tree_nodes[z_parent][TN_RIGHT_ID]:
                    # case 2
                    z = z_parent
                    # convert case 2 to case 3
                    root = _left_rotate(tree_vals, tree_nodes, root, z)
                # case 3
                z_parent = tree_nodes[z][TN_PARENT_ID]
                z_parent_parent = tree_nodes[z_parent][TN_PARENT_ID]
                tree_nodes[z_parent][TN_COLOR_ID] = RB_BLACK
                tree_nodes[z_parent_parent][TN_COLOR_ID] = RB_RED
                root = _right_rotate(tree_vals, tree_nodes, root,
                                     z_parent_parent)

        else:
            # (z->parent == z->parent->parent->right)
            y = tree_nodes[z_parent_parent][TN_LEFT_ID]
            if tree_nodes[y][TN_COLOR_ID] == RB_RED:
                # case 1
                tree_nodes[z_parent][TN_COLOR_ID] = RB_BLACK
                tree_nodes[y][TN_COLOR_ID] = RB_BLACK
                tree_nodes[z_parent_parent][TN_COLOR_ID] = RB_RED
                z = z_parent_parent
            else:
                if z == tree_nodes[z_parent][TN_LEFT_ID]:
                    # case 2
                    z = z_parent
                    # convert case 2 to case 3
                    root = _right_rotate(tree_vals, tree_nodes, root, z)
                # case 3
                z_parent = tree_nodes[z][TN_PARENT_ID]
                z_parent_parent = tree_nodes[z_parent][TN_PARENT_ID]
                tree_nodes[z_parent][TN_COLOR_ID] = RB_BLACK
                tree_nodes[z_parent_parent][TN_COLOR_ID] = RB_RED
                root = _left_rotate(tree_vals, tree_nodes, root,
                                    z_parent_parent)

        z_parent = tree_nodes[z][TN_PARENT_ID]

    tree_nodes[root][TN_COLOR_ID] = RB_BLACK
    return root


@jit(nb.i8(nb.f8[:, :], nb.i8[:, :], nb.i8, nb.i8, nb.f8[:]), nopython=True)
def _insert_into_tree(tree_vals, tree_nodes, root, node_id, value):
    # Create node and insert it into the tree
    cur_node = root

    if _compare(value[TN_KEY_ID], tree_vals[cur_node][TN_KEY_ID]) == -1:
        next_node = tree_nodes[cur_node][TN_LEFT_ID]
    else:
        next_node = tree_nodes[cur_node][TN_RIGHT_ID]

    while next_node != NIL_ID:
        cur_node = next_node
        if _compare(value[TN_KEY_ID], tree_vals[cur_node][TN_KEY_ID]) == -1:
            next_node = tree_nodes[cur_node][TN_LEFT_ID]
        else:
            next_node = tree_nodes[cur_node][TN_RIGHT_ID]

    # create a new node
    #   //and place it at the right place
    #   //created node is RED by default */
    _create_tree_nodes(tree_vals, tree_nodes, node_id, value, color=RB_RED)
    next_node = node_id

    tree_nodes[next_node][TN_PARENT_ID] = cur_node

    if _compare(value[TN_KEY_ID], tree_vals[cur_node][TN_KEY_ID]) == -1:
        tree_nodes[cur_node][TN_LEFT_ID] = next_node
    else:
        tree_nodes[cur_node][TN_RIGHT_ID] = next_node

    inserted = next_node

    # update augmented maxGradient
    tree_vals[next_node][TN_MAX_GRAD_ID] =\
        _find_value_min_value(tree_vals, next_node)

    while tree_nodes[next_node][TN_PARENT_ID] != NIL_ID:
        next_parent = tree_nodes[next_node][TN_PARENT_ID]
        if tree_vals[next_parent][TN_MAX_GRAD_ID] <\
                tree_vals[next_node][TN_MAX_GRAD_ID]:
            tree_vals[next_parent][TN_MAX_GRAD_ID] =\
                tree_vals[next_node][TN_MAX_GRAD_ID]

        if tree_vals[next_parent][TN_MAX_GRAD_ID] >\
                tree_vals[next_node][TN_MAX_GRAD_ID]:
            break

        next_node = next_parent

    # fix rb tree after insertion
    root = _rb_insert_fixup(tree_vals, tree_nodes, root, inserted)
    return root


@jit(nb.i8(nb.f8[:, :], nb.i8[:, :], nb.i8, nb.f8), nopython=True)
def _search_for_node(tree_vals, tree_nodes, root, key):
    # Search for a node with a given key.
    cur_node = root
    while cur_node != NIL_ID and \
            _compare(key, tree_vals[cur_node][TN_KEY_ID]) != 0:

        if _compare(key, tree_vals[cur_node][TN_KEY_ID]) == -1:
            cur_node = tree_nodes[cur_node][TN_LEFT_ID]
        else:
            cur_node = tree_nodes[cur_node][TN_RIGHT_ID]

    return cur_node


# The following is designed for viewshed's algorithm
@jit(nb.f8(nb.f8[:, :], nb.i8[:, :], nb.i8, nb.f8, nb.f8, nb.f8),
     nopython=True)
def _find_max_value_within_key(tree_vals, tree_nodes, root,
                               max_key, ang, gradient):
    key_node = _search_for_node(tree_vals, tree_nodes, root, max_key)
    if key_node == NIL_ID:
        # there is no point in the structure with key < maxKey */
        return SMALLEST_GRAD

    cur_node = key_node
    max = SMALLEST_GRAD
    while tree_nodes[cur_node][TN_PARENT_ID] != NIL_ID:
        cur_parent = tree_nodes[cur_node][TN_PARENT_ID]
        if cur_node == tree_nodes[cur_parent][TN_RIGHT_ID]:
            # its the right node of its parent
            cur_parent_left = tree_nodes[cur_parent][TN_LEFT_ID]
            tmp_max = _find_max_value(tree_vals[cur_parent_left])
            if tmp_max > max:
                max = tmp_max

            min_value = _find_value_min_value(tree_vals, cur_parent)
            if min_value > max:
                max = min_value

        cur_node = cur_parent

    if max > gradient:
        return max

    # traverse all nodes with smaller distance
    max = SMALLEST_GRAD
    cur_node = key_node
    while cur_node != NIL_ID:
        check_me = False
        if tree_vals[cur_node][TN_ANG_0] <= ang\
                <= tree_vals[cur_node][TN_ANG_2]:
            check_me = True
        if (not check_me) and tree_vals[cur_node][TN_KEY_ID] > 0:
            print('Angles outside angle')

        if tree_vals[cur_node][TN_KEY_ID] > max_key:
            raise ValueError("current dist too large ")

        if check_me and cur_node != key_node:

            if ang < tree_vals[cur_node][TN_ANG_1]:
                cur_grad = tree_vals[cur_node][TN_GRAD_1] \
                    + (tree_vals[cur_node][TN_GRAD_0]
                       - tree_vals[cur_node][TN_GRAD_1]) \
                    * (tree_vals[cur_node][TN_ANG_1] - ang) \
                    / (tree_vals[cur_node][TN_ANG_1]
                       - tree_vals[cur_node][TN_ANG_0])

            elif ang > tree_vals[cur_node][TN_ANG_1]:
                cur_grad = tree_vals[cur_node][TN_GRAD_1] \
                    + (tree_vals[cur_node][TN_GRAD_2]
                       - tree_vals[cur_node][TN_GRAD_1]) \
                    * (ang - tree_vals[cur_node][TN_ANG_1]) \
                    / (tree_vals[cur_node][TN_ANG_2]
                       - tree_vals[cur_node][TN_ANG_1])
            else:
                cur_grad = tree_vals[cur_node][TN_GRAD_1]

            if cur_grad > max:
                max = cur_grad

            if max > gradient:
                return max

        # get next smaller key
        if tree_nodes[cur_node][TN_LEFT_ID] != NIL_ID:
            cur_node = tree_nodes[cur_node][TN_LEFT_ID]
            while tree_nodes[cur_node][TN_RIGHT_ID] != NIL_ID:
                cur_node = tree_nodes[cur_node][TN_RIGHT_ID]
        else:
            # at smallest item in this branch, go back up
            last_node = cur_node
            cur_node = tree_nodes[cur_node][TN_PARENT_ID]
            while cur_node != NIL_ID and \
                    last_node == tree_nodes[cur_node][TN_LEFT_ID]:
                last_node = cur_node
                cur_node = tree_nodes[cur_node][TN_PARENT_ID]

    return max


@jit(nb.i8(nb.f8[:, :], nb.i8[:, :], nb.i8, nb.i8), nopython=True)
def _rb_delete_fixup(tree_vals, tree_nodes, root, x):
    # Fix the red-black tree after deletion.
    # This may change the root pointer.

    while x != root and tree_nodes[x][TN_COLOR_ID] == RB_BLACK:
        x_parent = tree_nodes[x][TN_PARENT_ID]
        if x == tree_nodes[x_parent][TN_LEFT_ID]:
            w = tree_nodes[x_parent][TN_RIGHT_ID]
            if tree_nodes[w][TN_COLOR_ID] == RB_RED:
                tree_nodes[w][TN_COLOR_ID] = RB_BLACK
                tree_nodes[x_parent][TN_COLOR_ID] = RB_RED
                root = _left_rotate(tree_vals, tree_nodes, root, x_parent)
                w = tree_nodes[x_parent][TN_RIGHT_ID]

            if w == NIL_ID:
                x = tree_nodes[x][TN_PARENT_ID]
                continue

            w_left = tree_nodes[w][TN_LEFT_ID]
            w_right = tree_nodes[w][TN_RIGHT_ID]
            if tree_nodes[w_left][TN_COLOR_ID] == RB_BLACK and \
                    tree_nodes[w_right][TN_COLOR_ID] == RB_BLACK:
                tree_nodes[w][TN_COLOR_ID] = RB_RED
                x = tree_nodes[x][TN_PARENT_ID]
            else:
                if tree_nodes[w_right][TN_COLOR_ID] == RB_BLACK:
                    tree_nodes[w_left][TN_COLOR_ID] = RB_BLACK
                    tree_nodes[w][TN_COLOR_ID] = RB_RED
                    root = _right_rotate(tree_vals, tree_nodes, root, w)
                    x_parent = tree_nodes[x][TN_PARENT_ID]
                    w = tree_nodes[x_parent][TN_RIGHT_ID]

                x_parent = tree_nodes[x][TN_PARENT_ID]
                w_right = tree_nodes[w][TN_RIGHT_ID]
                tree_nodes[w][TN_COLOR_ID] = tree_nodes[x_parent][TN_COLOR_ID]
                tree_nodes[x_parent][TN_COLOR_ID] = RB_BLACK
                tree_nodes[w_right][TN_COLOR_ID] = RB_BLACK
                root = _left_rotate(tree_vals, tree_nodes, root, x_parent)
                x = root
        else:
            # x == x.parent.right
            x_parent = tree_nodes[x][TN_PARENT_ID]
            w = tree_nodes[x_parent][TN_LEFT_ID]
            if tree_nodes[w][TN_COLOR_ID] == RB_RED:
                tree_nodes[w][TN_COLOR_ID] = RB_BLACK
                tree_nodes[x_parent][TN_COLOR_ID] = RB_RED
                root = _right_rotate(tree_vals, tree_nodes, root, x_parent)
                w = tree_nodes[x_parent][TN_LEFT_ID]

            if w == NIL_ID:
                x = x_parent
                continue

            w_left = tree_nodes[w][TN_LEFT_ID]
            w_right = tree_nodes[w][TN_RIGHT_ID]
            # do we need re-assignment here? No changes has been made for x?
            x_parent = tree_nodes[x][TN_PARENT_ID]
            if tree_nodes[w_right][TN_COLOR_ID] == RB_BLACK and \
                    tree_nodes[w_left][TN_COLOR_ID] == RB_BLACK:
                tree_nodes[w][TN_COLOR_ID] = RB_RED
                x = x_parent
            else:
                if tree_nodes[w_left][TN_COLOR_ID] == RB_BLACK:
                    tree_nodes[w_right][TN_COLOR_ID] = RB_BLACK
                    tree_nodes[w][TN_COLOR_ID] = RB_RED
                    root = _left_rotate(tree_vals, tree_nodes, root, w)
                    w = tree_nodes[x_parent][TN_LEFT_ID]
                tree_nodes[w][TN_COLOR_ID] = tree_nodes[x_parent][TN_COLOR_ID]
                tree_nodes[x_parent][TN_COLOR_ID] = RB_BLACK
                w_left = tree_nodes[w][TN_LEFT_ID]
                tree_nodes[w_left][TN_COLOR_ID] = RB_BLACK
                root = _right_rotate(tree_vals, tree_nodes, root, x_parent)
                x = root

    tree_nodes[x][TN_COLOR_ID] = RB_BLACK
    return root


@jit(nb.types.Tuple((nb.i8, nb.i8))(nb.f8[:, :], nb.i8[:, :], nb.i8, nb.f8),
     nopython=True)
def _delete_from_tree(tree_vals, tree_nodes, root, key):
    # Delete the node out of the tree. This may change the root pointer.

    z = _search_for_node(tree_vals, tree_nodes, root, key)

    if z == NIL_ID:
        # node to delete is not found
        raise ValueError("node not found")

    # 1-3
    if tree_nodes[z][TN_LEFT_ID] == NIL_ID or\
            tree_nodes[z][TN_RIGHT_ID] == NIL_ID:
        y = z
    else:
        y = _tree_successor(tree_nodes, z)

    if y == NIL_ID:
        raise ValueError("successor not found")

    deleted = y

    # 4-6
    if tree_nodes[y][TN_LEFT_ID] != NIL_ID:
        x = tree_nodes[y][TN_LEFT_ID]
    else:
        x = tree_nodes[y][TN_RIGHT_ID]

    # 7
    tree_nodes[x][TN_PARENT_ID] = tree_nodes[y][TN_PARENT_ID]

    # 8-12
    if tree_nodes[y][TN_PARENT_ID] == NIL_ID:
        root = x
        # augmentation to be fixed
        to_fix = root
    else:
        y_parent = tree_nodes[y][TN_PARENT_ID]
        if y == tree_nodes[y_parent][TN_LEFT_ID]:
            tree_nodes[y_parent][TN_LEFT_ID] = x
        else:
            tree_nodes[y_parent][TN_RIGHT_ID] = x
        # augmentation to be fixed
        to_fix = y_parent

    # fix augmentation for removing y
    cur_node = y

    while tree_nodes[cur_node][TN_PARENT_ID] != NIL_ID:
        cur_parent = tree_nodes[cur_node][TN_PARENT_ID]
        if tree_vals[cur_parent][TN_MAX_GRAD_ID] == \
                _find_value_min_value(tree_vals, y):
            cur_parent_left = tree_nodes[cur_parent][TN_LEFT_ID]
            cur_parent_right = tree_nodes[cur_parent][TN_RIGHT_ID]
            left = _find_max_value(tree_vals[cur_parent_left])
            right = _find_max_value(tree_vals[cur_parent_right])

            if left > right:
                tree_vals[cur_parent][TN_MAX_GRAD_ID] = left
            else:
                tree_vals[cur_parent][TN_MAX_GRAD_ID] = right

            min_value = _find_value_min_value(tree_vals, cur_parent)
            if min_value > tree_vals[cur_parent][TN_MAX_GRAD_ID]:
                tree_vals[cur_parent][TN_MAX_GRAD_ID] = min_value

        else:
            break

        cur_node = cur_parent

    # fix augmentation for x
    to_fix_left = tree_nodes[to_fix][TN_LEFT_ID]
    to_fix_right = tree_nodes[to_fix][TN_RIGHT_ID]
    if tree_vals[to_fix_left][TN_MAX_GRAD_ID] >\
            tree_vals[to_fix_right][TN_MAX_GRAD_ID]:
        tmp_max = tree_vals[to_fix_left][TN_MAX_GRAD_ID]
    else:
        tmp_max = tree_vals[to_fix_right][TN_MAX_GRAD_ID]

    min_value = _find_value_min_value(tree_vals, to_fix)
    if tmp_max > min_value:
        tree_vals[to_fix][TN_MAX_GRAD_ID] = tmp_max
    else:
        tree_vals[to_fix][TN_MAX_GRAD_ID] = min_value

    # 13-15
    if y != NIL_ID and y != z:
        z_gradient = _find_value_min_value(tree_vals, z)
        tree_vals[z][TN_KEY_ID] = tree_vals[y][TN_KEY_ID]
        tree_vals[z][TN_GRAD_0] = tree_vals[y][TN_GRAD_0]
        tree_vals[z][TN_GRAD_1] = tree_vals[y][TN_GRAD_1]
        tree_vals[z][TN_GRAD_2] = tree_vals[y][TN_GRAD_2]
        tree_vals[z][TN_ANG_0] = tree_vals[y][TN_ANG_0]
        tree_vals[z][TN_ANG_1] = tree_vals[y][TN_ANG_1]
        tree_vals[z][TN_ANG_2] = tree_vals[y][TN_ANG_2]

        to_fix = z
        # fix augmentation
        to_fix_left = tree_nodes[to_fix][TN_LEFT_ID]
        to_fix_right = tree_nodes[to_fix][TN_RIGHT_ID]
        if tree_vals[to_fix_left][TN_MAX_GRAD_ID] > \
                tree_vals[to_fix_right][TN_MAX_GRAD_ID]:
            tmp_max = tree_vals[to_fix_left][TN_MAX_GRAD_ID]
        else:
            tmp_max = tree_vals[to_fix_right][TN_MAX_GRAD_ID]

        min_value = _find_value_min_value(tree_vals, to_fix)
        if tmp_max > min_value:
            tree_vals[to_fix][TN_MAX_GRAD_ID] = tmp_max
        else:
            tree_vals[to_fix][TN_MAX_GRAD_ID] = min_value

        while tree_nodes[z][TN_PARENT_ID] != NIL_ID:
            z_parent = tree_nodes[z][TN_PARENT_ID]
            if tree_vals[z_parent][TN_MAX_GRAD_ID] == z_gradient:
                z_parent_left = tree_nodes[z_parent][TN_LEFT_ID]
                z_parent_right = tree_nodes[z_parent][TN_RIGHT_ID]
                x_parent = tree_nodes[x][TN_PARENT_ID]
                x_parent_right = tree_nodes[x_parent][TN_RIGHT_ID]
                if _find_value_min_value(tree_vals, z_parent) != z_gradient\
                    and \
                    not (tree_vals[z_parent_left][TN_MAX_GRAD_ID] == z_gradient
                         and
                         tree_vals[x_parent_right][TN_MAX_GRAD_ID] ==
                         z_gradient):

                    left = _find_max_value(tree_vals[z_parent_left])
                    right = _find_max_value(tree_vals[z_parent_right])

                    if left > right:
                        tree_vals[z_parent][TN_MAX_GRAD_ID] = left
                    else:
                        tree_vals[z_parent][TN_MAX_GRAD_ID] = right

                    min_value = _find_value_min_value(tree_vals, z_parent)
                    if min_value > tree_vals[z_parent][TN_MAX_GRAD_ID]:
                        tree_vals[z_parent][TN_MAX_GRAD_ID] = min_value

            else:
                if tree_vals[z][TN_MAX_GRAD_ID] >\
                        tree_vals[z_parent][TN_MAX_GRAD_ID]:
                    tree_vals[z_parent][TN_MAX_GRAD_ID] =\
                        tree_vals[z][TN_MAX_GRAD_ID]

            z = z_parent

    # 16-17
    if tree_nodes[y][TN_COLOR_ID] == RB_BLACK and x != NIL_ID:
        root = _rb_delete_fixup(tree_vals, tree_nodes, root, x)

    # 18
    return root, deleted


def _print_status_node(sn, row, col):
    print("row=", row, "col=", col, "dist_to_viewpoint=",
          sn[TN_KEY_ID], "grad=", sn[TN_GRAD_0], sn[TN_GRAD_1], sn[TN_GRAD_2],
          "ang=", sn[TN_ANG_0], sn[TN_ANG_1], sn[TN_ANG_2])
    return


@jit(nb.f8(nb.f8[:, :], nb.i8[:, :], nb.i8, nb.f8, nb.f8, nb.f8),
     nopython=True)
def _max_grad_in_status_struct(tree_vals, tree_nodes, root,
                               distance, angle, gradient):
    # Find the node with max Gradient within the distance (from vp)
    # Note: if there is nothing in the status structure,
    #         it means this cell is VISIBLE

    if root == NIL_ID:
        return SMALLEST_GRAD

    # it is also possible that the status structure is not empty, but
    # there are no events with key < dist ---in this case it returns
    # SMALLEST_GRAD;

    # find max within the max key

    return _find_max_value_within_key(tree_vals, tree_nodes, root,
                                      distance, angle, gradient)


@jit(nb.f8(nb.f8, nb.f8, nb.f8), nopython=True)
def _col_to_east(col, window_west, window_ew_res):
    # Column to easting.
    # Converts a column relative to a window to an east coordinate.
    return window_west + col * window_ew_res


@jit(nb.f8(nb.f8, nb.f8, nb.f8), nopython=True)
def _row_to_north(row, window_north, window_ns_res):
    # Row to northing.
    # Converts a row relative to a window to an north coordinate.
    return window_north - row * window_ns_res


@jit(nb.f8(nb.f8), nopython=True)
def _radian(x):
    # Convert degree into radian.
    return x * PI / 180.0


@jit(nb.f8(nb.f8, nb.f8), nopython=True)
def _hypot(x, y):
    return sqrt(x * x + y * y)


@jit(nb.f8(nb.f8, nb.f8, nb.f8, nb.f8), nopython=True)
def _g_distance(e1, n1, e2, n2):
    # Computes the distance, in meters, from (x1, y1) to (x2, y2)

    # assume meter grid
    factor = 1.0
    return factor * _hypot(e1 - e2, n1 - n2)


@jit(nb.void(nb.f8[:, :], nb.i8, nb.i8, nb.f8), nopython=True)
def _set_visibility(visibility_grid, i, j, value):
    visibility_grid[i][j] = value
    return


@jit(nb.types.Tuple((nb.i8, nb.i8))(nb.i8, nb.i8, nb.i8,
                                    nb.i8, nb.i8), nopython=True)
def _calculate_event_row_col(event_type, event_row, event_col,
                             viewpoint_row, viewpoint_col):
    # Calculate the neighbouring of the given event.
    x = 0
    y = 0
    if event_type == CENTER_EVENT:
        raise ValueError("_calculate_event_row_col() must not be called for "
                         "CENTER events")

    if event_row < viewpoint_row and event_col < viewpoint_col:
        # first quadrant
        if event_type == ENTERING_EVENT:
            # if it is ENTERING_EVENT
            y = event_row - 1
            x = event_col + 1
        else:
            # if it is EXITING_EVENT
            y = event_row + 1
            x = event_col - 1

    elif event_col == viewpoint_col and event_row < viewpoint_row:
        # between the first and second quadrant
        if event_type == ENTERING_EVENT:
            # if it is ENTERING_EVENT
            y = event_row + 1
            x = event_col + 1
        else:
            # if it is EXITING_EVENT
            y = event_row + 1
            x = event_col - 1

    elif event_col > viewpoint_col and event_row < viewpoint_row:
        # second quadrant
        if event_type == ENTERING_EVENT:
            # if it is ENTERING_EVENT
            y = event_row + 1
            x = event_col + 1
        else:
            # if it is EXITING_EVENT
            y = event_row - 1
            x = event_col - 1

    elif event_col > viewpoint_col and event_row == viewpoint_row:
        # between the second and forth quadrant
        if event_type == ENTERING_EVENT:
            # if it is ENTERING_EVENT
            y = event_row + 1
            x = event_col - 1
        else:
            # if it is EXITING_EVENT
            y = event_row - 1
            x = event_col - 1

    elif event_col > viewpoint_col and event_row > viewpoint_row:
        # forth quadrant
        if event_type == ENTERING_EVENT:
            # if it is ENTERING_EVENT
            y = event_row + 1
            x = event_col - 1
        else:
            # if it is EXITING_EVENT
            y = event_row - 1
            x = event_col + 1

    elif event_col == viewpoint_col and event_row > viewpoint_row:
        # between the third and fourth quadrant
        if event_type == ENTERING_EVENT:
            # if it is ENTERING_EVENT
            y = event_row - 1
            x = event_col - 1
        else:
            # if it is EXITING_EVENT
            y = event_row - 1
            x = event_col + 1

    elif event_col < viewpoint_col and event_row > viewpoint_row:
        # third quadrant
        if event_type == ENTERING_EVENT:
            # if it is ENTERING_EVENT
            y = event_row - 1
            x = event_col - 1
        else:
            # if it is EXITING_EVENT
            y = event_row + 1
            x = event_col + 1

    elif event_col < viewpoint_col and event_row == viewpoint_row:
        # between the first and third quadrant
        if event_type == ENTERING_EVENT:
            # if it is ENTERING_EVENT
            y = event_row - 1
            x = event_col + 1
        else:
            # if it is EXITING_EVENT
            y = event_row + 1
            x = event_col + 1

    else:
        # must be the vp cell itself
        assert event_row == viewpoint_row and event_col == viewpoint_col
        x = event_col
        y = event_row

    if abs(x - event_col > 1) or abs(y - event_row > 1):
        raise ValueError("_calculate_event_row_col()")

    return y, x


@jit(nb.f8(nb.i8, nb.i8, nb.i8, nb.i8, nb.i8,
           nb.i8, nb.i8, nb.f8[:, :]), nopython=True)
def _calc_event_elev(event_type, event_row, event_col, n_rows, n_cols,
                     viewpoint_row, viewpoint_col, inrast):
    # Calculate ENTER and EXIT event elevation (bilinear interpolation)

    row1, col1 = _calculate_event_row_col(event_type, event_row, event_col,
                                          viewpoint_row, viewpoint_col)

    event_elev = inrast[1][event_col]

    if 0 <= row1 < n_rows and 0 <= col1 < n_cols:
        elev1 = inrast[row1 - event_row + 1][col1]
        elev2 = inrast[row1 - event_row + 1][event_col]
        elev3 = inrast[1][col1]
        elev4 = inrast[1][event_col]
        if np.isnan(elev1) or np.isnan(elev2) or np.isnan(elev3) \
                or np.isnan(elev4):
            event_elev = inrast[1][event_col]
        else:
            event_elev = (elev1 + elev2 + elev3 + elev4) / 4.0

    return event_elev


@jit(nb.types.Tuple((nb.f8, nb.f8))(nb.i8, nb.i8, nb.i8,
                                    nb.i8, nb.i8), nopython=True)
def _calc_event_pos(event_type, event_row, event_col,
                    viewpoint_row, viewpoint_col):
    # Calculate the exact position of the given event,
    # and store them in x and y.

    # Quadrants:  1 2
    #   3 4
    #   ----->x
    #   |
    #   |
    #   |
    #   V y

    x = 0
    y = 0
    if event_type == CENTER_EVENT:
        # FOR CENTER_EVENTS
        y = event_row
        x = event_col
        return y, x

    if event_row < viewpoint_row and event_col < viewpoint_col:
        # first quadrant
        if event_type == ENTERING_EVENT:
            # if it is ENTERING_EVENT
            y = event_row - 0.5
            x = event_col + 0.5
        else:
            # if it is EXITING_EVENT
            y = event_row + 0.5
            x = event_col - 0.5

    elif event_row < viewpoint_row and event_col == viewpoint_col:
        # between the first and second quadrant
        if event_type == ENTERING_EVENT:
            # if it is ENTERING_EVENT
            y = event_row + 0.5
            x = event_col + 0.5
        else:
            # if it is EXITING_EVENT
            y = event_row + 0.5
            x = event_col - 0.5

    elif event_row < viewpoint_row and event_col > viewpoint_col:
        # second quadrant
        if event_type == ENTERING_EVENT:
            # if it is ENTERING_EVENT
            y = event_row + 0.5
            x = event_col + 0.5
        else:
            # if it is EXITING_EVENT
            y = event_row - 0.5
            x = event_col - 0.5

    elif event_row == viewpoint_row and event_col > viewpoint_col:
        # between the second and the fourth quadrant
        if event_type == ENTERING_EVENT:
            # if it is ENTERING_EVENT
            y = event_row + 0.5
            x = event_col - 0.5
        else:
            # if it is EXITING_EVENT
            y = event_row - 0.5
            x = event_col - 0.5

    elif event_row > viewpoint_row and event_col > viewpoint_col:
        # fourth quadrant
        if event_type == ENTERING_EVENT:
            # if it is ENTERING_EVENT
            y = event_row + 0.5
            x = event_col - 0.5
        else:
            # if it is EXITING_EVENT
            y = event_row - 0.5
            x = event_col + 0.5

    elif event_row > viewpoint_row and event_col == viewpoint_col:
        # between the third and fourth quadrant
        if event_type == ENTERING_EVENT:
            # if it is ENTERING_EVENT
            y = event_row - 0.5
            x = event_col - 0.5
        else:
            # if it is EXITING_EVENT
            y = event_row - 0.5
            x = event_col + 0.5

    elif event_row > viewpoint_row and event_col < viewpoint_col:
        # third quadrant
        if event_type == ENTERING_EVENT:
            # if it is ENTERING_EVENT
            y = event_row - 0.5
            x = event_col - 0.5
        else:
            # if it is EXITING_EVENT
            y = event_row + 0.5
            x = event_col + 0.5

    elif event_row == viewpoint_row and event_col < viewpoint_col:
        # between first and third quadrant
        if event_type == ENTERING_EVENT:
            # if it is ENTERING_EVENT
            y = event_row - 0.5
            x = event_col + 0.5
        else:
            # if it is EXITING_EVENT
            y = event_row + 0.5
            x = event_col + 0.5

    else:
        # must be the vp cell itself
        assert event_row == viewpoint_row and event_col == viewpoint_col
        x = event_col
        y = event_row

    assert abs(event_col - x) < 1 and abs(event_row - y) < 1

    return y, x


@jit(nb.f8(nb.f8, nb.f8, nb.i8, nb.i8), nopython=True)
def _calculate_angle(event_x, event_y, viewpoint_x, viewpoint_y):
    if viewpoint_x == event_x and viewpoint_y > event_y:
        # between 1st and 2nd quadrant
        return PI / 2

    if viewpoint_x == event_x and viewpoint_y < event_y:
        # between 3rd and 4th quadrant
        return PI * 3.0 / 2.0

    # Calculate angle between (x1, y1) and (x2, y2)
    ang = atan(fabs(event_y - viewpoint_y) / fabs(event_x - viewpoint_x))

    # M_PI is defined in math.h to represent 3.14159...
    if viewpoint_y == event_y and event_x > viewpoint_x:
        # between 1st and 4th quadrant
        return 0

    if event_x > viewpoint_x and event_y < viewpoint_y:
        # first quadrant
        return ang

    if viewpoint_x > event_x and viewpoint_y > event_y:
        # 2nd quadrant
        return PI - ang

    if viewpoint_x > event_x and viewpoint_y == event_y:
        # between 1st and 3rd quadrant
        return PI

    if viewpoint_x > event_x and viewpoint_y < event_y:
        # 3rd quadrant
        return PI + ang

    if viewpoint_x < event_x and viewpoint_y < event_y:
        # 4th quadrant
        return PI * 2.0 - ang

    assert event_x == viewpoint_x and event_y == viewpoint_y
    return 0


@jit(nb.f8(nb.f8, nb.f8, nb.f8, nb.i8, nb.i8,
           nb.f8, nb.f8, nb.f8), nopython=True)
def _calc_event_grad(row, col, elev, viewpoint_row, viewpoint_col,
                     viewpoint_elev, ew_res, ns_res):
    # Calculate event gradient

    diff_elev = elev - viewpoint_elev

    dx = (col - viewpoint_col) * ew_res
    dy = (row - viewpoint_row) * ns_res
    distance_to_viewpoint = (dx * dx) + (dy * dy)

    # PI / 2 above, - PI / 2 below
    if distance_to_viewpoint == 0:
        if diff_elev > 0:
            gradient = PI / 2
        elif diff_elev < 0:
            gradient = - PI / 2
        else:
            gradient = 0
    else:
        gradient = atan(diff_elev / sqrt(distance_to_viewpoint))
    return gradient


# given a StatusNode, fill in its dist2vp and gradient
@jit(nb.types.Tuple((nb.f8, nb.f8))(nb.i8, nb.i8, nb.f8, nb.i8,
                                    nb.i8, nb.f8, nb.f8, nb.f8),
     nopython=True)
def _calc_dist_n_grad(status_node_row, status_node_col, elev, viewpoint_row,
                      viewpoint_col, viewpoint_elev, ew_res, ns_res):
    diff_elev = elev - viewpoint_elev

    dx = (status_node_col - viewpoint_col) * ew_res
    dy = (status_node_row - viewpoint_row) * ns_res
    distance_to_viewpoint = (dx * dx) + (dy * dy)

    # PI / 2 above, - PI / 2 below
    if distance_to_viewpoint == 0:
        if diff_elev > 0:
            gradient = PI / 2
        elif diff_elev < 0:
            gradient = - PI / 2
        else:
            gradient = 0
    else:
        gradient = atan(diff_elev / sqrt(distance_to_viewpoint))
    return distance_to_viewpoint, gradient


# ported https://github.com/OSGeo/grass/blob/master/raster/r.viewshed/grass.cpp
# function _init_event_list_in_memory()
@jit(nb.void(nb.f8[:, :], nb.f8[:, :], nb.i8, nb.i8,
             nb.f8[:, :], nb.f8[:, :]), nopython=True)
def _init_event_list(event_list, raster, vp_row, vp_col,
                     data, visibility_grid):
    # Initialize and fill all the events for the map into event_list

    n_rows, n_cols = raster.shape
    inrast = np.empty(shape=(3, n_cols), dtype=np.float64)
    inrast.fill(np.nan)

    # scan through the raster data
    # read first row
    inrast[2] = raster[0]

    # index of the event array: row, col, elev_0, elev_1, elev_2, ang, type
    e = np.zeros((7,), dtype=np.float64)

    count_event = 0
    for i in range(n_rows):
        # read in the raster row
        tmprast = inrast[0]
        inrast[0] = inrast[1]
        inrast[1] = inrast[2]
        inrast[2] = tmprast

        if i < n_rows - 1:
            inrast[2] = raster[i + 1]
        else:
            for j in range(n_cols):
                # when assign to None, it is forced to np.nan
                inrast[2][j] = np.nan

        # fill event list with events from this row
        for j in range(n_cols):
            # integer
            e_row = i
            e_col = j

            # float
            e[E_ROW_ID] = i
            e[E_COL_ID] = j

            # read the elevation value into the event
            e[E_ELEV_1] = inrast[1][j]

            # write it into the row of data going through the vp
            if i == vp_row:
                data[0][j] = e[E_ELEV_1]
                data[1][j] = e[E_ELEV_1]
                data[2][j] = e[E_ELEV_1]

            # set the vp, and don't insert it into eventlist
            if i == vp_row and j == vp_col:
                _set_visibility(visibility_grid, i, j, 180)
                continue

            # if it got here it is not the vp, not NODATA, and
            # within max distance from vp generate its 3 events
            # and insert them

            # get ENTER elevation
            e[E_TYPE_ID] = ENTERING_EVENT
            e[E_ELEV_0] = _calc_event_elev(e[E_TYPE_ID], e_row, e_col,
                                           n_rows, n_cols,
                                           vp_row, vp_col, inrast)

            # get EXIT event
            e[E_TYPE_ID] = EXITING_EVENT
            e[E_ELEV_2] = _calc_event_elev(e[E_TYPE_ID], e_row, e_col,
                                           n_rows, n_cols,
                                           vp_row, vp_col, inrast)

            # write adjusted elevation into the row of data
            # going through the vp
            if i == vp_row:
                data[0][j] = e[E_ELEV_0]
                data[1][j] = e[E_ELEV_1]
                data[2][j] = e[E_ELEV_2]

            # put event into event list
            e[E_TYPE_ID] = ENTERING_EVENT
            ay, ax = _calc_event_pos(e[E_TYPE_ID], e_row, e_col,
                                     vp_row, vp_col)
            e[E_ANG_ID] = _calculate_angle(ax, ay, vp_col, vp_row)
            event_list[count_event] = e
            count_event += 1

            e[E_TYPE_ID] = CENTER_EVENT
            ay, ax = _calc_event_pos(e[E_TYPE_ID], e_row, e_col,
                                     vp_row, vp_col)
            e[E_ANG_ID] = _calculate_angle(ax, ay, vp_col, vp_row)
            event_list[count_event] = e
            count_event += 1

            e[E_TYPE_ID] = EXITING_EVENT
            ay, ax = _calc_event_pos(e[E_TYPE_ID], e_row, e_col,
                                     vp_row, vp_col)
            e[E_ANG_ID] = _calculate_angle(ax, ay, vp_col, vp_row)
            event_list[count_event] = e
            count_event += 1

    return


@jit(nb.i8(nb.f8[:, :], nb.i8[:, :]), nopython=True)
def _create_status_struct(tree_vals, tree_nodes):
    # Create and initialize the status struct.
    # return a Tree object with a dummy root.

    # dummy status node
    dummy_node_value = np.array([0.0, -1, -1, SMALLEST_GRAD, SMALLEST_GRAD,
                                SMALLEST_GRAD, 0.0, 0.0, 0.0, SMALLEST_GRAD])

    # node 0 is root
    root = 0
    _create_tree_nodes(tree_vals, tree_nodes, root, dummy_node_value, RB_BLACK)

    # last row is NIL
    _create_tree_nodes(tree_vals, tree_nodes, NIL_ID,
                       dummy_node_value, RB_BLACK)

    num_nodes = tree_vals.shape[0]
    tree_nodes[NIL_ID][TN_LEFT_ID] = num_nodes
    tree_nodes[NIL_ID][TN_RIGHT_ID] = num_nodes
    tree_nodes[NIL_ID][TN_PARENT_ID] = num_nodes

    return root


# /*find the vertical ang in degrees between the vp and the
#    point represented by the StatusNode.  Assumes all values (except
#    gradient) in sn have been filled. The value returned is in [0,
#    180]. A value of 0 is directly below the specified viewing position,
#    90 is due horizontal, and 180 is directly above the observer.
#    If doCurv is set we need to consider the curvature of the
#    earth */
@jit(nb.f8(nb.f8, nb.f8, nb.f8), nopython=True)
def _get_vertical_ang(viewpoint_elev, distance_to_viewpoint, elev):
    # Find the vertical angle in degrees between the vp
    # and the point represented by the StatusNode

    # determine the difference in elevation, based on the curvature
    diff_elev = viewpoint_elev - elev

    # calculate and return the ang in degrees
    assert abs(distance_to_viewpoint) > 0.0

    # 0 above, 180 below
    if diff_elev == 0.0:
        return 90
    elif diff_elev > 0:
        return atan(sqrt(distance_to_viewpoint) / diff_elev) * 180 / PI

    return atan(abs(diff_elev) / sqrt(distance_to_viewpoint)) * 180 / PI + 90


@jit(nb.void(nb.f8[:]), nopython=True)
def _init_status_node(status_node):
    status_node[TN_KEY_ID] = -1

    status_node[TN_GRAD_0] = np.nan
    status_node[TN_GRAD_1] = np.nan
    status_node[TN_GRAD_2] = np.nan

    status_node[TN_ANG_0] = np.nan
    status_node[TN_ANG_1] = np.nan
    status_node[TN_ANG_2] = np.nan

    return


def _print_event(event):
    if event[E_TYPE_ID] == 1:
        t = "ENTERING   "
    elif event[E_TYPE_ID] == -1:
        t = "EXITING    "
    else:
        t = "CENTER     "

    print('row = ', event[E_ROW_ID],
          'col = ', event[E_COL_ID],
          'event_type = ', t,
          'elevation = ', event[E_ELEV_0], event[E_ELEV_1], event[E_ELEV_2],
          'ang = ', event[E_ANG_ID])
    return


@jit(nb.void(nb.i8[:], nb.i8), nopython=True)
def _push(stack, item):
    stack[0] += 1
    stack[stack[0]] = item
    return


@jit(nb.i8(nb.i8[:]), nopython=True)
def _pop(stack):
    item = stack[stack[0]]
    stack[0] -= 1
    return item


# Viewshed's sweep algorithm on the grid stored in the given file, and
# with the given vp.  Create a visibility grid and return
# it. The computation runs in memory, which means the input grid, the
# status structure and the output grid are stored in arrays in
# memory.
#
# The output: A cell x in the visibility grid is recorded as follows:
#
# if it is NODATA, then x  is set to NODATA
# if it is invisible, then x is set to INVISIBLE
# if it is visible,  then x is set to the vertical ang wrt to vp

# https://github.com/OSGeo/grass/blob/master/raster/r.viewshed/viewshed.cpp
# function viewshed_in_memory()

@jit(nb.f8[:, :](nb.f8[:, :], nb.i8, nb.i8, nb.f8, nb.f8, nb.f8, nb.f8,
                 nb.i8[:, :], nb.f8[:, :], nb.f8[:, :], nb.f8[:, :]),
     nopython=True)
def _viewshed(raster, vp_row, vp_col, vp_elev, vp_target, ew_res, ns_res,
              event_rcts, event_aes, data, visibility_grid):
    n_rows, n_cols = raster.shape

    # for e in event_list:
    #     _print_event(e)

    # create the status structure
    # create 2d array of the RB-tree
    num_nodes = n_cols - vp_col + n_cols * n_rows + 10

    status_values = np.zeros((num_nodes, 8), dtype=np.float64)
    status_struct = np.zeros((num_nodes, 4), dtype=np.int64)

    root = _create_status_struct(status_values, status_struct)

    # idle row idx in the 2d data array of status_struct tree
    idle = np.zeros((num_nodes,), dtype=np.int64)
    for i in range(0, num_nodes - 1):
        idle[i] = num_nodes - i
    idle[0] = num_nodes - 2

    # Put cells that are initially on the sweepline into status structure
    status_node = np.zeros((7,), dtype=np.float64)
    for i in range(vp_col + 1, n_cols):
        _init_status_node(status_node)
        status_row = vp_row
        status_col = i

        # event properties
        e_row = vp_row
        e_col = i
        e_elev_0 = data[0][i]
        e_elev_1 = data[1][i]
        e_elev_2 = data[2][i]

        if (not np.isnan(data[1][i])):
            # calculate Distance to VP and Gradient,
            # store them into status_node
            # need either 3 elevation values or
            # 3 gradients calculated from 3 elevation values
            # need also 3 angs

            e_type = ENTERING_EVENT
            ay, ax = _calc_event_pos(e_type, e_row, e_col, vp_row, vp_col)
            status_node[TN_ANG_0] = _calculate_angle(ax, ay, vp_col, vp_row)
            status_node[TN_GRAD_0] = _calc_event_grad(ay, ax, e_elev_0,
                                                      vp_row, vp_col, vp_elev,
                                                      ew_res, ns_res)

            e_type = CENTER_EVENT
            ay, ax = _calc_event_pos(e_type, e_row, e_col, vp_row, vp_col)
            status_node[TN_ANG_1] = _calculate_angle(ax, ay, vp_col, vp_row)
            status_node[TN_KEY_ID], status_node[TN_GRAD_1] = \
                _calc_dist_n_grad(status_row, status_col, e_elev_1,
                                  vp_row, vp_col, vp_elev, ew_res, ns_res)

            e_type = EXITING_EVENT
            ay, ax = _calc_event_pos(e_type, e_row, e_col, vp_row, vp_col)
            status_node[TN_ANG_2] = _calculate_angle(ax, ay, vp_col, vp_row)
            status_node[TN_GRAD_2] = _calc_event_grad(ay, ax, e_elev_2,
                                                      vp_row, vp_col, vp_elev,
                                                      ew_res, ns_res)

            assert status_node[TN_ANG_1] == 0

            if status_node[TN_ANG_0] > status_node[TN_ANG_1]:
                status_node[TN_ANG_0] -= 2 * PI

            # insert sn into the status structure
            id = _pop(idle)
            root = _insert_into_tree(status_values, status_struct, root,
                                     id, status_node)

    # sweep the event_list

    nevents = len(event_rcts)

    for i in range(nevents):
        # get out one event at a time and process it according to its type
        e_rct = event_rcts[i]
        e_ae = event_aes[i]
        # e = event_list[i]

        # status_node = StatusNode(row=e[E_ROW_ID], col=e[E_COL_ID])
        _init_status_node(status_node)
        status_row = e_rct[E_ROW_ID]
        status_col = e_rct[E_COL_ID]

        # calculate Distance to VP and Gradient
        status_node[TN_KEY_ID], status_node[TN_GRAD_1] = \
            _calc_dist_n_grad(status_row, status_col,
                              e_ae[AE_ELEV_1] + vp_target,
                              vp_row, vp_col, vp_elev, ew_res, ns_res,)

        etype = e_rct[E_TYPE_ID]
        if etype == ENTERING_EVENT:
            # insert node into structure

            #  need either 3 elevation values or
            # 	     * 3 gradients calculated from 3 elevation values */
            # 	    /* need also 3 angs */
            ay, ax = _calc_event_pos(e_rct[E_TYPE_ID], e_rct[E_ROW_ID],
                                     e_rct[E_COL_ID], vp_row, vp_col)
            status_node[TN_ANG_0] = e_ae[AE_ANG_ID]
            status_node[TN_GRAD_0] = _calc_event_grad(ay, ax, e_ae[AE_ELEV_0],
                                                      vp_row, vp_col, vp_elev,
                                                      ew_res, ns_res)

            e_rct[E_TYPE_ID] = CENTER_EVENT
            ay, ax = _calc_event_pos(e_rct[E_TYPE_ID], e_rct[E_ROW_ID],
                                     e_rct[E_COL_ID], vp_row, vp_col)
            status_node[TN_ANG_1] = _calculate_angle(ax, ay, vp_col, vp_row)
            status_node[TN_KEY_ID], status_node[TN_GRAD_1] = \
                _calc_dist_n_grad(status_row, status_col, e_ae[AE_ELEV_1],
                                  vp_row, vp_col, vp_elev, ew_res, ns_res)

            e_rct[E_TYPE_ID] = EXITING_EVENT
            ay, ax = _calc_event_pos(e_rct[E_TYPE_ID], e_rct[E_ROW_ID],
                                     e_rct[E_COL_ID], vp_row, vp_col)
            status_node[TN_ANG_2] = _calculate_angle(ax, ay, vp_col, vp_row)
            status_node[TN_GRAD_2] = _calc_event_grad(ay, ax, e_ae[AE_ELEV_2],
                                                      vp_row, vp_col, vp_elev,
                                                      ew_res, ns_res)

            e_rct[E_TYPE_ID] = ENTERING_EVENT

            if e_ae[AE_ANG_ID] < PI:
                if status_node[TN_ANG_0] > status_node[TN_ANG_1]:
                    status_node[TN_ANG_0] -= 2 * PI
            else:
                if status_node[TN_ANG_0] > status_node[TN_ANG_1]:
                    status_node[TN_ANG_1] += 2 * PI
                    status_node[TN_ANG_2] += 2 * PI

            id = _pop(idle)
            root = _insert_into_tree(status_values, status_struct, root,
                                     id, status_node)

        elif etype == EXITING_EVENT:
            # delete node out of status structure
            root, deleted = _delete_from_tree(status_values, status_struct,
                                              root, status_node[TN_KEY_ID])
            _push(idle, deleted)

        elif etype == CENTER_EVENT:
            # calculate visibility
            # consider current ang and gradient
            max = _max_grad_in_status_struct(status_values, status_struct,
                                             root, status_node[TN_KEY_ID],
                                             e_ae[AE_ANG_ID],
                                             status_node[TN_GRAD_1])

            # the point is visible: store its vertical ang
            if max <= status_node[TN_GRAD_1]:
                vert_ang = _get_vertical_ang(vp_elev, status_node[TN_KEY_ID],
                                             e_ae[AE_ELEV_1] + vp_target)

                _set_visibility(visibility_grid, status_row,
                                status_col, vert_ang)

                assert vert_ang >= 0
                # when you write the visibility grid you assume that
                # 		   visible values are positive

    return visibility_grid


def viewshed(raster, x, y, observer_elev=OBS_ELEV, target_elev=TARGET_ELEV):
    """Calculate viewshed of a raster (the visible cells in the raster)
    for the given viewpoint (observer) location.

    Parameters
    ----------
    raster: xarray.DataArray
        Input raster image.
    x: int, float
        x-coordinate in data space of observer location
    y: int, float
        y-coordinate in data space of observer location
    observer_elev: float
        Observer elevation above the terrain.
    target_elev: float
        Target elevation offset above the terrain.


    Returns
    -------
    viewshed: xarray.DataArray
             A cell x in the visibility grid is recorded as follows:
                If it is invisible, then x is set to INVISIBLE.
                If it is visible,  then x is set to the vertical angle w.r.t
                the viewpoint.
    """
    height, width = raster.shape

    y_coords = raster.indexes.get('y').values
    x_coords = raster.indexes.get('x').values

    # validate x arg
    if x < x_coords[0]:
        raise ValueError("x argument outside of raster x_range")
    elif x > x_coords[-1]:
        raise ValueError("x argument outside of raster x_range")

    # validate y arg
    if y < y_coords[0]:
        raise ValueError("y argument outside of raster y_range")
    elif y > y_coords[-1]:
        raise ValueError("y argument outside of raster y_range")

    selection = raster.sel(x=[x], y=[y], method='nearest')
    x = selection.x.values[0]
    y = selection.y.values[0]

    y_view = np.where(y_coords == y)[0][0]
    y_range = (y_coords[0], y_coords[-1])

    x_view = np.where(x_coords == x)[0][0]
    x_range = (x_coords[0], x_coords[-1])

    # viewpoint properties
    viewpoint_row = y_view
    viewpoint_col = x_view
    viewpoint_elev = raster.values[y_view, x_view] + observer_elev
    viewpoint_target = 0.0
    if target_elev > 0:
        viewpoint_target = target_elev

    # int getgrdhead(FILE * fd, struct Cell_head *cellhd)
    ew_res = (x_range[1] - x_range[0]) / (width - 1)
    ns_res = (y_range[1] - y_range[0]) / (height - 1)

    # create the visibility grid of the sizes specified in the header
    visibility_grid = np.empty(shape=raster.shape, dtype=np.float64)
    # set everything initially invisible
    visibility_grid.fill(INVISIBLE)
    n_rows, n_cols = raster.shape

    data = np.zeros(shape=(3, n_cols), dtype=np.float64)

    # construct the event list corresponding to the given input file and vp;
    # this creates an array of all the cells on the same row as the vp
    num_events = 3 * (n_rows * n_cols - 1)
    event_list = np.zeros((num_events, 7), dtype=np.float64)

    raster.values = raster.values.astype(np.float64)

    _init_event_list(event_list=event_list, raster=raster.values,
                     vp_row=viewpoint_row, vp_col=viewpoint_col,
                     data=data, visibility_grid=visibility_grid)

    # sort the events radially by ang
    event_list = event_list[np.lexsort((event_list[:, E_TYPE_ID],
                                        event_list[:, E_ANG_ID]))]

    # event indices: row, col, type, ang, enter elev, center elev, exit elev
    # split event into 2 arrays: one of 3 integer elements: row, col, type;
    #                          and one of 4 float elements: angle, elevations.
    event_rcts = np.array(event_list[:, :3], dtype=np.int64)
    event_aes = np.array(event_list[:, 3:], dtype=np.float64)

    viewshed_img = _viewshed(raster.values, viewpoint_row, viewpoint_col,
                             viewpoint_elev, viewpoint_target, ew_res, ns_res,
                             event_rcts, event_aes, data, visibility_grid)

    visibility = xarray.DataArray(viewshed_img,
                                  coords=raster.coords,
                                  attrs=raster.attrs,
                                  dims=raster.dims)
    return visibility
