import xarray
import numpy as np
import math

from math import atan, sqrt, fabs
from math import pi as PI

import numba as nb
from numba import jit

E_ROW_ID = 0
E_COL_ID = 1
E_ELVEV_0 = 2
E_ELVEV_1 = 3
E_ELVEV_2 = 4
E_ANG_ID = 5
E_TYPE_ID = 6

S_DIST2VP = 0
S_ROW_ID = 1
S_COL_ID = 2
S_GRAD_0 = 3
S_GRAD_1 = 4
S_GRAD_2 = 5
S_ANG_0 = 6
S_ANG_1 = 7
S_ANG_2 = 8

TN_KEY_ID = 0
TN_GRAD_0 = 1
TN_GRAD_1 = 2
TN_GRAD_2 = 3
TN_ANG_0 = 4
TN_ANG_1 = 5
TN_ANG_2 = 6
TN_MAX_GRAD_ID = 7
TN_COLOR_ID = 8
TN_LEFT_ID = 9
TN_RIGHT_ID = 10
TN_PARENT_ID = 11

NIL_ID = -1

VP_ROW_ID = 0
VP_COL_ID = 1
VP_ELEV_ID = 2
VP_TARGET_ID = 3

VO_OBS_ELEV_ID = 0
VO_TARGET_ID = 1
VO_MAX_DIST_ID = 2
VO_CURVE_ID = 3
VO_REFR_ID = 4
VO_ELLPS_A_ID = 5
VO_REFR_COEF_ID = 6

GH_PROJ_ID = 0
GH_EW_RES_ID = 1
GH_NS_RES_ID = 2
GH_NORTH_ID = 3
GH_SOUTH_ID = 4
GH_EAST_ID = 5
GH_WEST_ID = 6

# view options default values
OBS_ELEV = 0
TARGET_ELEV = 0
ELLPS_A = 6370997.0
DO_CURVE = False
DO_REFR = False
REFR_COEF = 1.0 / 7.0

# max distance default value
INF = -1

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
SMALLEST_GRAD = float('-inf')

PROJ_LL = 0
PROJ_NONE = -1

NAN = -9999999999999999


@jit(nb.i8(nb.f8, nb.f8), nopython=True)
def _compare(a, b):
    if a < b:
        return -1
    if a > b:
        return 1
    return 0


@jit(nb.f8(nb.f8[:, :], nb.i8), nopython=True)
def _find_value_min_value(tree, node_id):
    return min(tree[node_id][TN_GRAD_0],
               tree[node_id][TN_GRAD_1],
               tree[node_id][TN_GRAD_2])


def _print_tree(status_struct):
    for i in range(len(status_struct)):
        print(i, status_struct[i][0])


def _print_tv(tv):
    print('key=', tv[TN_KEY_ID],
          'grad=', tv[TN_GRAD_0], tv[TN_GRAD_1], tv[TN_GRAD_2],
          'ang=', tv[TN_ANG_0], tv[TN_ANG_1], tv[TN_ANG_2],
          'max_grad=', tv[TN_MAX_GRAD_ID])
    return


@jit(nb.void(nb.f8[:, :], nb.i8, nb.f8[:], nb.i8), nopython=True)
def _create_tree_node(tree, x, val, color=RB_RED):
    # Create a TreeNode using given TreeValue

    # every node has null nodes as children initially, create one such object
    # for easy management

    tree[x][TN_KEY_ID] = val[S_DIST2VP]
    tree[x][TN_GRAD_0] = val[S_GRAD_0]
    tree[x][TN_GRAD_1] = val[S_GRAD_1]
    tree[x][TN_GRAD_2] = val[S_GRAD_2]
    tree[x][TN_ANG_0] = val[S_ANG_0]
    tree[x][TN_ANG_1] = val[S_ANG_1]
    tree[x][TN_ANG_2] = val[S_ANG_2]
    tree[x][TN_MAX_GRAD_ID] = SMALLEST_GRAD
    tree[x][TN_COLOR_ID] = color
    tree[x][TN_LEFT_ID] = NIL_ID
    tree[x][TN_RIGHT_ID] = NIL_ID
    tree[x][TN_PARENT_ID] = NIL_ID
    return


@jit(nb.i8(nb.f8[:, :], nb.i8), nopython=True)
def _tree_minimum(tree, x):
    while int(tree[x][TN_LEFT_ID]) != NIL_ID:
        x = int(tree[x][TN_LEFT_ID])
    return x


# function used by deletion
@jit(nb.i8(nb.f8[:, :], nb.i8), nopython=True)
def _tree_successor(tree, x):
    # Find the highest successor of a node in the tree

    if tree[x][TN_RIGHT_ID] != NIL_ID:
        return _tree_minimum(tree, int(tree[x][TN_RIGHT_ID]))

    y = int(tree[x][TN_PARENT_ID])
    while y != NIL_ID and x == int(tree[y][TN_RIGHT_ID]):
        x = y
        if tree[y][TN_PARENT_ID] == NIL_ID:
            return y
        y = int(tree[y][TN_PARENT_ID])
    return y


@jit(nb.f8(nb.f8[:]), nopython=True)
def _find_max_value(node_value):
    # Find the max value in the given tree.
    return node_value[TN_MAX_GRAD_ID]


@jit(nb.i8(nb.f8[:, :], nb.i8, nb.i8), nopython=True)
def _left_rotate(tree, root, x):
    # A utility function to left rotate subtree rooted with a node.

    y = int(tree[x][TN_RIGHT_ID])

    # fix x
    x_left = int(tree[x][TN_LEFT_ID])
    y_left = int(tree[y][TN_LEFT_ID])
    if tree[x_left][TN_MAX_GRAD_ID] > tree[y_left][TN_MAX_GRAD_ID]:
        tmp_max = tree[x_left][TN_MAX_GRAD_ID]
    else:
        tmp_max = tree[y_left][TN_MAX_GRAD_ID]

    min_value = _find_value_min_value(tree, x)
    if tmp_max > min_value:
        tree[x][TN_MAX_GRAD_ID] = tmp_max
    else:
        tree[x][TN_MAX_GRAD_ID] = min_value

    # fix y
    y_right = int(tree[y][TN_RIGHT_ID])
    if tree[x][TN_MAX_GRAD_ID] > tree[y_right][TN_MAX_GRAD_ID]:
        tmp_max = tree[x][TN_MAX_GRAD_ID]
    else:
        tmp_max = tree[y_right][TN_MAX_GRAD_ID]

    min_value = _find_value_min_value(tree, y)
    if tmp_max > min_value:
        tree[y][TN_MAX_GRAD_ID] = tmp_max
    else:
        tree[y][TN_MAX_GRAD_ID] = min_value

    # left rotation
    # see pseudo code on page 278 CLRS

    # turn y's left subtree into x's right subtree
    tree[x][TN_RIGHT_ID] = tree[y][TN_LEFT_ID]
    y_left = int(tree[y][TN_LEFT_ID])
    tree[y_left][TN_PARENT_ID] = x
    # link x's parent to y
    tree[y][TN_PARENT_ID] = int(tree[x][TN_PARENT_ID])

    if tree[x][TN_PARENT_ID] == NIL_ID:
        root = y
    else:
        x_parent = int(tree[x][TN_PARENT_ID])
        if x == int(tree[x_parent][TN_LEFT_ID]):
            tree[x_parent][TN_LEFT_ID] = y
        else:
            tree[x_parent][TN_RIGHT_ID] = y

    tree[y][TN_LEFT_ID] = x
    tree[x][TN_PARENT_ID] = y
    return root


@jit(nb.i8(nb.f8[:, :], nb.i8, nb.i8), nopython=True)
def _right_rotate(tree, root, y):
    # A utility function to right rotate subtree rooted with a node.

    x = int(tree[y][TN_LEFT_ID])

    # fix y
    x_right = int(tree[x][TN_RIGHT_ID])
    y_right = int(tree[y][TN_RIGHT_ID])
    if tree[x_right][TN_MAX_GRAD_ID] > tree[y_right][TN_MAX_GRAD_ID]:
        tmp_max = tree[x_right][TN_MAX_GRAD_ID]
    else:
        tmp_max = tree[y_right][TN_MAX_GRAD_ID]

    min_value = _find_value_min_value(tree, y)
    if tmp_max > min_value:
        tree[y][TN_MAX_GRAD_ID] = tmp_max
    else:
        tree[y][TN_MAX_GRAD_ID] = min_value

    # fix x
    x_left = int(tree[x][TN_LEFT_ID])
    if tree[x_left][TN_MAX_GRAD_ID] > tree[y][TN_MAX_GRAD_ID]:
        tmp_max = tree[x_left][TN_MAX_GRAD_ID]
    else:
        tmp_max = tree[y][TN_MAX_GRAD_ID]

    min_value = _find_value_min_value(tree, x)
    if tmp_max > min_value:
        tree[x][TN_MAX_GRAD_ID] = tmp_max
    else:
        tree[x][TN_MAX_GRAD_ID] = min_value

    # rotation
    tree[y][TN_LEFT_ID] = tree[x][TN_RIGHT_ID]
    x_right = int(tree[x][TN_RIGHT_ID])
    tree[x_right][TN_PARENT_ID] = y

    tree[x][TN_PARENT_ID] = tree[y][TN_PARENT_ID]

    if tree[y][TN_PARENT_ID] == NIL_ID:
        root = x
    else:
        y_parent = int(tree[y][TN_PARENT_ID])
        if tree[y_parent][TN_LEFT_ID] == y:
            tree[y_parent][TN_LEFT_ID] = x
        else:
            tree[y_parent][TN_RIGHT_ID] = x

    tree[x][TN_RIGHT_ID] = y
    tree[y][TN_PARENT_ID] = x
    return root


@jit(nb.i8(nb.f8[:, :], nb.i8, nb.i8), nopython=True)
def _rb_insert_fixup(tree, root, z):
    # Fix red-black tree after insertion. This may change the root pointer.

    # see pseudocode on page 281 in CLRS
    z_parent = int(tree[z][TN_PARENT_ID])
    while tree[z_parent][TN_COLOR_ID] == RB_RED:
        z_parent_parent = int(tree[z_parent][TN_PARENT_ID])
        if tree[z][TN_PARENT_ID] == tree[z_parent_parent][TN_LEFT_ID]:
            y = int(tree[z_parent_parent][TN_RIGHT_ID])
            if tree[y][TN_COLOR_ID] == RB_RED:
                # case 1
                tree[z_parent][TN_COLOR_ID] = RB_BLACK
                tree[y][TN_COLOR_ID] = RB_BLACK
                tree[z_parent_parent][TN_COLOR_ID] = RB_RED
                # re assignment for z
                z = z_parent_parent
            else:
                if z == int(tree[z_parent][TN_RIGHT_ID]):
                    # case 2
                    z = z_parent
                    # convert case 2 to case 3
                    root = _left_rotate(tree, root, z)
                # case 3
                z_parent = int(tree[z][TN_PARENT_ID])
                z_parent_parent = int(tree[z_parent][TN_PARENT_ID])
                tree[z_parent][TN_COLOR_ID] = RB_BLACK
                tree[z_parent_parent][TN_COLOR_ID] = RB_RED
                root = _right_rotate(tree, root, z_parent_parent)

        else:
            # (z->parent == z->parent->parent->right)
            y = int(tree[z_parent_parent][TN_LEFT_ID])
            if tree[y][TN_COLOR_ID] == RB_RED:
                # case 1
                tree[z_parent][TN_COLOR_ID] = RB_BLACK
                tree[y][TN_COLOR_ID] = RB_BLACK
                tree[z_parent_parent][TN_COLOR_ID] = RB_RED
                z = z_parent_parent
            else:
                if z == int(tree[z_parent][TN_LEFT_ID]):
                    # case 2
                    z = z_parent
                    # convert case 2 to case 3
                    root = _right_rotate(tree, root, z)
                # case 3
                z_parent = int(tree[z][TN_PARENT_ID])
                z_parent_parent = int(tree[z_parent][TN_PARENT_ID])
                tree[z_parent][TN_COLOR_ID] = RB_BLACK
                tree[z_parent_parent][TN_COLOR_ID] = RB_RED
                root = _left_rotate(tree, root, z_parent_parent)

        z_parent = int(tree[z][TN_PARENT_ID])

    tree[root][TN_COLOR_ID] = RB_BLACK
    return root


@jit(nb.i8(nb.f8[:, :], nb.i8, nb.i8, nb.f8[:]), nopython=True)
def _insert_into_tree(tree, root, node_id, value):
    # Create node and insert it into the tree
    cur_node = root

    if _compare(value[TN_KEY_ID], tree[cur_node][TN_KEY_ID]) == -1:
        next_node = int(tree[cur_node][TN_LEFT_ID])
    else:
        next_node = int(tree[cur_node][TN_RIGHT_ID])

    while next_node != NIL_ID:
        cur_node = next_node
        if _compare(value[TN_KEY_ID], tree[cur_node][TN_KEY_ID]) == -1:
            next_node = int(tree[cur_node][TN_LEFT_ID])
        else:
            next_node = int(tree[cur_node][TN_RIGHT_ID])

    # create a new node
    #   //and place it at the right place
    #   //created node is RED by default */
    _create_tree_node(tree, node_id, value, color=RB_RED)
    next_node = node_id

    tree[next_node][TN_PARENT_ID] = cur_node

    if _compare(value[TN_KEY_ID], tree[cur_node][TN_KEY_ID]) == -1:
        tree[cur_node][TN_LEFT_ID] = next_node
    else:
        tree[cur_node][TN_RIGHT_ID] = next_node

    inserted = next_node

    # update augmented maxGradient
    tree[next_node][TN_MAX_GRAD_ID] = _find_value_min_value(tree, next_node)
    while tree[next_node][TN_PARENT_ID] != NIL_ID:
        next_parent = int(tree[next_node][TN_PARENT_ID])
        if tree[next_parent][TN_MAX_GRAD_ID] < tree[next_node][TN_MAX_GRAD_ID]:
            tree[next_parent][TN_MAX_GRAD_ID] = tree[next_node][TN_MAX_GRAD_ID]

        if tree[next_parent][TN_MAX_GRAD_ID] > tree[next_node][TN_MAX_GRAD_ID]:
            break

        next_node = next_parent

    # fix rb tree after insertion
    root = _rb_insert_fixup(tree, root, inserted)
    return root


@jit(nb.i8(nb.f8[:, :], nb.i8, nb.f8), nopython=True)
def _search_for_node(tree, root, key):
    # Search for a node with a given key.
    cur_node = root
    while cur_node != NIL_ID and \
            _compare(key, tree[cur_node][TN_KEY_ID]) != 0:

        if _compare(key, tree[cur_node][TN_KEY_ID]) == -1:
            cur_node = int(tree[cur_node][TN_LEFT_ID])
        else:
            cur_node = int(tree[cur_node][TN_RIGHT_ID])

    return cur_node


# The following is designed for viewshed's algorithm
@jit(nb.f8(nb.f8[:, :], nb.i8, nb.f8, nb.f8, nb.f8), nopython=True)
def _find_max_value_within_key(tree, root, max_key, ang, gradient):
    key_node = _search_for_node(tree, root, max_key)
    if key_node == NIL_ID:
        # there is no point in the structure with key < maxKey */
        return SMALLEST_GRAD

    cur_node = key_node
    max = SMALLEST_GRAD
    while tree[cur_node][TN_PARENT_ID] != NIL_ID:
        cur_parent = int(tree[cur_node][TN_PARENT_ID])
        if cur_node == int(tree[cur_parent][TN_RIGHT_ID]):
            # its the right node of its parent
            cur_parent_left = int(tree[cur_parent][TN_LEFT_ID])
            tmp_max = _find_max_value(tree[cur_parent_left])
            if tmp_max > max:
                max = tmp_max

            min_value = _find_value_min_value(tree, cur_parent)
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
        if tree[cur_node][TN_ANG_0] <= ang <= tree[cur_node][TN_ANG_2]:
            check_me = True
        if (not check_me) and tree[cur_node][TN_KEY_ID] > 0:
            print('Angles outside angle')

        if tree[cur_node][TN_KEY_ID] > max_key:
            raise ValueError("current dist too large ")

        if check_me and cur_node != key_node:

            if ang < tree[cur_node][TN_ANG_1]:
                cur_grad = tree[cur_node][TN_GRAD_1] \
                           + (tree[cur_node][TN_GRAD_0] - tree[cur_node][
                            TN_GRAD_1]) \
                           * (tree[cur_node][TN_ANG_1] - ang) \
                           / (tree[cur_node][TN_ANG_1] - tree[cur_node][
                            TN_ANG_0])

            elif ang > tree[cur_node][TN_ANG_1]:
                cur_grad = tree[cur_node][TN_GRAD_1] \
                           + (tree[cur_node][TN_GRAD_2] - tree[cur_node][
                            TN_GRAD_1]) \
                           * (ang - tree[cur_node][TN_ANG_1]) \
                           / (tree[cur_node][TN_ANG_2] - tree[cur_node][
                            TN_ANG_1])
            else:
                cur_grad = tree[cur_node][TN_GRAD_1]

            if cur_grad > max:
                max = cur_grad

            if max > gradient:
                return max

        # get next smaller key
        if tree[cur_node][TN_LEFT_ID] != NIL_ID:
            cur_node = int(tree[cur_node][TN_LEFT_ID])
            while tree[cur_node][TN_RIGHT_ID] != NIL_ID:
                cur_node = int(tree[cur_node][TN_RIGHT_ID])
        else:
            # at smallest item in this branch, go back up
            last_node = cur_node
            cur_node = int(tree[cur_node][TN_PARENT_ID])
            while cur_node != NIL_ID and \
                    last_node == int(tree[cur_node][TN_LEFT_ID]):
                last_node = cur_node
                cur_node = int(tree[cur_node][TN_PARENT_ID])

    return max


@jit(nb.i8(nb.f8[:, :], nb.i8, nb.i8), nopython=True)
def _rb_delete_fixup(tree, root, x):
    # Fix the red-black tree after deletion.
    # This may change the root pointer.

    while x != root and tree[x][TN_COLOR_ID] == RB_BLACK:
        x_parent = int(tree[x][TN_PARENT_ID])
        if x == int(tree[x_parent][TN_LEFT_ID]):
            w = int(tree[x_parent][TN_RIGHT_ID])
            if tree[w][TN_COLOR_ID] == RB_RED:
                tree[w][TN_COLOR_ID] = RB_BLACK
                tree[x_parent][TN_COLOR_ID] = RB_RED
                root = _left_rotate(tree, root, x_parent)
                w = int(tree[x_parent][TN_RIGHT_ID])

            if w == NIL_ID:
                x = int(tree[x][TN_PARENT_ID])
                continue

            w_left = int(tree[w][TN_LEFT_ID])
            w_right = int(tree[w][TN_RIGHT_ID])
            if tree[w_left][TN_COLOR_ID] == RB_BLACK and \
                    tree[w_right][TN_COLOR_ID] == RB_BLACK:
                tree[w][TN_COLOR_ID] = RB_RED
                x = int(tree[x][TN_PARENT_ID])
            else:
                if tree[w_right][TN_COLOR_ID] == RB_BLACK:
                    tree[w_left][TN_COLOR_ID] = RB_BLACK
                    tree[w][TN_COLOR_ID] = RB_RED
                    root = _right_rotate(tree, root, w)
                    x_parent = int(tree[x][TN_PARENT_ID])
                    w = int(tree[x_parent][TN_RIGHT_ID])

                x_parent = int(tree[x][TN_PARENT_ID])
                w_right = int(tree[w][TN_RIGHT_ID])
                tree[w][TN_COLOR_ID] = tree[x_parent][TN_COLOR_ID]
                tree[x_parent][TN_COLOR_ID] = RB_BLACK
                tree[w_right][TN_COLOR_ID] = RB_BLACK
                root = _left_rotate(tree, root, x_parent)
                x = root
        else:
            # x == x.parent.right
            x_parent = int(tree[x][TN_PARENT_ID])
            w = int(tree[x_parent][TN_LEFT_ID])
            if tree[w][TN_COLOR_ID] == RB_RED:
                tree[w][TN_COLOR_ID] = RB_BLACK
                tree[x_parent][TN_COLOR_ID] = RB_RED
                root = _right_rotate(tree, root, x_parent)
                w = int(tree[x_parent][TN_LEFT_ID])

            if w == NIL_ID:
                x = x_parent
                continue

            w_left = int(tree[w][TN_LEFT_ID])
            w_right = int(tree[w][TN_RIGHT_ID])
            # do we need re-assignment here? No changes has been made for x?
            x_parent = int(tree[x][TN_PARENT_ID])
            if tree[w_right][TN_COLOR_ID] == RB_BLACK and \
                    tree[w_left][TN_COLOR_ID] == RB_BLACK:
                tree[w][TN_COLOR_ID] = RB_RED
                x = x_parent
            else:
                if tree[w_left][TN_COLOR_ID] == RB_BLACK:
                    tree[w_right][TN_COLOR_ID] = RB_BLACK
                    tree[w][TN_COLOR_ID] = RB_RED
                    root = _left_rotate(tree, root, w)
                    w = int(tree[x_parent][TN_LEFT_ID])
                tree[w][TN_COLOR_ID] = tree[x_parent][TN_COLOR_ID]
                tree[x_parent][TN_COLOR_ID] = RB_BLACK
                w_left = int(tree[w][TN_LEFT_ID])
                tree[w_left][TN_COLOR_ID] = RB_BLACK
                root = _right_rotate(tree, root, x_parent)
                x = root

    tree[x][TN_COLOR_ID] = RB_BLACK
    return root


@jit(nb.types.Tuple((nb.i8, nb.i8))(nb.f8[:, :], nb.i8, nb.f8), nopython=True)
def _delete_from_tree(tree, root, key):
    # Delete the node out of the tree. This may change the root pointer.

    z = _search_for_node(tree, root, key)

    if z == NIL_ID:
        # node to delete is not found
        raise ValueError("node not found")

    # 1-3
    if tree[z][TN_LEFT_ID] == NIL_ID or tree[z][TN_RIGHT_ID] == NIL_ID:
        y = z
    else:
        y = _tree_successor(tree, z)

    if y == NIL_ID:
        raise ValueError("successor not found")

    deleted = y

    # 4-6
    if tree[y][TN_LEFT_ID] != NIL_ID:
        x = int(tree[y][TN_LEFT_ID])
    else:
        x = int(tree[y][TN_RIGHT_ID])

    # 7
    tree[x][TN_PARENT_ID] = tree[y][TN_PARENT_ID]

    # 8-12
    if tree[y][TN_PARENT_ID] == NIL_ID:
        root = x
        # augmentation to be fixed
        to_fix = root
    else:
        y_parent = int(tree[y][TN_PARENT_ID])
        if y == int(tree[y_parent][TN_LEFT_ID]):
            tree[y_parent][TN_LEFT_ID] = x
        else:
            tree[y_parent][TN_RIGHT_ID] = x
        # augmentation to be fixed
        to_fix = y_parent

    # fix augmentation for removing y
    cur_node = y

    while tree[cur_node][TN_PARENT_ID] != NIL_ID:
        cur_parent = int(tree[cur_node][TN_PARENT_ID])
        if tree[cur_parent][TN_MAX_GRAD_ID] == \
                _find_value_min_value(tree, y):
            cur_parent_left = int(tree[cur_parent][TN_LEFT_ID])
            cur_parent_right = int(tree[cur_parent][TN_RIGHT_ID])
            left = _find_max_value(tree[cur_parent_left])
            right = _find_max_value(tree[cur_parent_right])

            if left > right:
                tree[cur_parent][TN_MAX_GRAD_ID] = left
            else:
                tree[cur_parent][TN_MAX_GRAD_ID] = right

            min_value = _find_value_min_value(tree, cur_parent)
            if min_value > tree[cur_parent][TN_MAX_GRAD_ID]:
                tree[cur_parent][TN_MAX_GRAD_ID] = min_value

        else:
            break

        cur_node = cur_parent

    # fix augmentation for x
    to_fix_left = int(tree[to_fix][TN_LEFT_ID])
    to_fix_right = int(tree[to_fix][TN_RIGHT_ID])
    if tree[to_fix_left][TN_MAX_GRAD_ID] > tree[to_fix_right][TN_MAX_GRAD_ID]:
        tmp_max = tree[to_fix_left][TN_MAX_GRAD_ID]
    else:
        tmp_max = tree[to_fix_right][TN_MAX_GRAD_ID]

    min_value = _find_value_min_value(tree, to_fix)
    if tmp_max > min_value:
        tree[to_fix][TN_MAX_GRAD_ID] = tmp_max
    else:
        tree[to_fix][TN_MAX_GRAD_ID] = min_value

    # 13-15
    if y != NIL_ID and y != z:
        z_gradient = _find_value_min_value(tree, z)
        tree[z][TN_KEY_ID] = tree[y][TN_KEY_ID]
        tree[z][TN_GRAD_0] = tree[y][TN_GRAD_0]
        tree[z][TN_GRAD_1] = tree[y][TN_GRAD_1]
        tree[z][TN_GRAD_2] = tree[y][TN_GRAD_2]
        tree[z][TN_ANG_0] = tree[y][TN_ANG_0]
        tree[z][TN_ANG_1] = tree[y][TN_ANG_1]
        tree[z][TN_ANG_2] = tree[y][TN_ANG_2]

        to_fix = z
        # fix augmentation
        to_fix_left = int(tree[to_fix][TN_LEFT_ID])
        to_fix_right = int(tree[to_fix][TN_RIGHT_ID])
        if tree[to_fix_left][TN_MAX_GRAD_ID] > \
                tree[to_fix_right][TN_MAX_GRAD_ID]:
            tmp_max = tree[to_fix_left][TN_MAX_GRAD_ID]
        else:
            tmp_max = tree[to_fix_right][TN_MAX_GRAD_ID]

        min_value = _find_value_min_value(tree, to_fix)
        if tmp_max > min_value:
            tree[to_fix][TN_MAX_GRAD_ID] = tmp_max
        else:
            tree[to_fix][TN_MAX_GRAD_ID] = min_value

        while tree[z][TN_PARENT_ID] != NIL_ID:
            z_parent = int(tree[z][TN_PARENT_ID])
            if tree[z_parent][TN_MAX_GRAD_ID] == z_gradient:
                z_parent_left = int(tree[z_parent][TN_LEFT_ID])
                z_parent_right = int(tree[z_parent][TN_RIGHT_ID])
                x_parent = int(tree[x][TN_PARENT_ID])
                x_parent_right = int(tree[x_parent][TN_RIGHT_ID])
                if _find_value_min_value(tree, z_parent) != z_gradient and \
                        not (tree[z_parent_left][TN_MAX_GRAD_ID] == z_gradient
                             and tree[x_parent_right][
                                 TN_MAX_GRAD_ID] == z_gradient):

                    left = _find_max_value(tree[z_parent_left])
                    right = _find_max_value(tree[z_parent_right])

                    if left > right:
                        tree[z_parent][TN_MAX_GRAD_ID] = left
                    else:
                        tree[z_parent][TN_MAX_GRAD_ID] = right

                    min_value = _find_value_min_value(tree, z_parent)
                    if min_value > tree[z_parent][TN_MAX_GRAD_ID]:
                        tree[z_parent][TN_MAX_GRAD_ID] = min_value

            else:
                if tree[z][TN_MAX_GRAD_ID] > tree[z_parent][TN_MAX_GRAD_ID]:
                    tree[z_parent][TN_MAX_GRAD_ID] = tree[z][TN_MAX_GRAD_ID]

            z = z_parent

    # 16-17
    if tree[y][TN_COLOR_ID] == RB_BLACK and x != NIL_ID:
        root = _rb_delete_fixup(tree, root, x)

    # 18
    return root, deleted


def _print_status_node(sn):
    print("row=", sn[S_ROW_ID], "col=", sn[S_COL_ID], "dist_to_viewpoint=",
          sn[S_DIST2VP], "grad=", sn[S_GRAD_0], sn[S_GRAD_1], sn[S_GRAD_2],
          "ang=", sn[S_ANG_0], sn[S_ANG_1], sn[S_ANG_2])
    return


@jit(nb.f8(nb.f8[:, :], nb.i8, nb.f8, nb.f8, nb.f8), nopython=True)
def _max_grad_in_status_struct(tree, root, distance, angle, gradient):
    # Find the node with max Gradient within the distance (from vp)
    # Note: if there is nothing in the status structure,
    #         it means this cell is VISIBLE

    if root == NIL_ID:
        return SMALLEST_GRAD

    # it is also possible that the status structure is not empty, but
    # there are no events with key < dist ---in this case it returns
    # SMALLEST_GRAD;

    # find max within the max key

    return _find_max_value_within_key(tree, root, distance, angle, gradient)


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


# TODO Move this to utils or proximity.py
@jit(nb.f8(nb.f8, nb.f8, nb.f8, nb.f8), nopython=True)
def _g_geodesic_distance(lon1, lat1, lon2, lat2):
    # Calculates geodesic distance from (lon1, lat1) to (lon2, lat2) in meters.
    # G_begin_distance_calculations

    # G_get_ellipsoid_parameters
    a = 6378137.0
    e2 = .006694385
    al = a
    boa = sqrt(1 - e2)
    f = 1 - boa
    ff64 = f * f / 64

    # #define Radians(x) ((x) * PI/180.0)
    # G_set_geodesic_distance_lat1
    t1r = atan(boa * math.tan(_radian(lat1)))

    # G_set_geodesic_distance_lat2
    t2r = atan(boa * math.tan(_radian(lat2)))
    tm = (t1r + t2r) / 2
    dtm = (t2r - t1r) / 2

    stm = math.sin(tm)
    ctm = math.cos(tm)
    sdtm = math.sin(dtm)
    cdtm = math.cos(dtm)

    t1 = stm * cdtm
    t1 = t1 * t1 * 2

    t2 = sdtm * ctm
    t2 = t2 * t2 * 2

    t3 = sdtm * sdtm
    t4 = cdtm * cdtm - stm * stm

    # _g_geodesic_distance_lon_to_lon
    sdlmr = math.sin(_radian((lon2 - lon1) / 2))

    # special case - shapiro
    if sdlmr == 0.0 and t1r == t2r:
        return 0

    q = t3 + sdlmr * sdlmr * t4
    # special case - shapiro

    if q == 1:
        return PI * al

    # /* Mod: shapiro
    # * cd=1-2q is ill-conditioned if q is small O(10**-23)
    # *   (for high lats? with lon1-lon2 < .25 degrees?)
    # *   the computation of cd = 1-2*q will give cd==1.0.
    # * However, note that t=dl/sd is dl/sin(dl) which approaches 1 as dl->0.
    # * So the first step is to compute a good value for sd without using sin()
    # *   and then check cd && q to see if we got cd==1.0 when we shouldn't.
    # * Note that dl isn't used except to get t,
    # *   but both cd and sd are used later
    # */
    #
    # /* original code
    #   cd=1-2*q;
    #   dl=acos(cd);
    #   sd=sin(dl);
    #   t=dl/sd;
    # */

    cd = 1 - 2 * q  # ill-conditioned subtraction for small q
    # mod starts here
    sd = 2 * sqrt(q - q * q)  # sd^2 = 1 - cd^2
    if q != 0.0 and cd == 1.0:
        t = 1.0
    elif sd == 0.0:
        t = 1.0
    else:
        t = math.acos(cd) / sd  # don't know how to fix acos(1-2*q) yet
    # mod ends here

    u = t1 / (1 - q)
    v = t2 / q
    d = 4 * t * t
    x = u + v
    e = -2 * cd
    y = u - v
    a = -d * e

    return al * sd * (t - f / 4 * (t * x - y)
                      + ff64 * (x * (a + (t - (a + e) / 2) * x)
                                + y * (-2 * d + e * y) + d * x * y))


@jit(nb.f8(nb.f8, nb.f8), nopython=True)
def _hypot(x, y):
    return sqrt(x * x + y * y)


@jit(nb.f8(nb.f8, nb.f8, nb.f8, nb.f8, nb.i8), nopython=True)
def _g_distance(e1, n1, e2, n2, proj=PROJ_NONE):
    # Computes the distance, in meters, from (x1, y1) to (x2, y2)

    if proj == PROJ_LL:
        return _g_geodesic_distance(e1, n1, e2, n2)

    else:
        # assume meter grid
        factor = 1.0
        return factor * _hypot(e1 - e2, n1 - n2)


# If viewOptions.doCurv is on then adjust the passed height for
# curvature of the earth; otherwise return the passed height
# unchanged.
# If viewOptions.doRefr is on then adjust the curved height for
# the effect of atmospheric refraction too.
@jit(nb.f8(nb.i8, nb.i8, nb.f8, nb.f8, nb.f8, nb.b1, nb.f8,
           nb.b1, nb.f8, nb.f8, nb.f8, nb.f8, nb.f8, nb.i8), nopython=True)
def _adjust_curv(viewpoint_row, viewpoint_col, row, col, h, do_curv, ellps_a,
                 do_refr, refr_coef, west, ew_res, north, ns_res, proj):
    # Adjust the passed height for curvature of the earth
    #     and the effect of atmospheric refraction.

    if not do_curv:
        return h

    assert ellps_a != 0

    # distance must be in meters because ellps_a is in meters
    # _col_to_east(col, window_west, window_ew_res)
    # _row_to_north(row, window_north, window_ns_res)

    dist = _g_distance(_col_to_east(viewpoint_col + 0.5, west, ew_res),
                       _row_to_north(viewpoint_row + 0.5, north, ns_res),
                       _col_to_east(col + 0.5, west, ew_res),
                       _row_to_north(row + 0.5, north, ns_res),
                       proj)

    adjustment = dist * dist / (2 * ellps_a)

    if not do_refr:
        return h - adjustment

    return h - (adjustment * (1.0 - refr_coef))


@jit(nb.void(nb.f8[:, :], nb.i8, nb.i8, nb.f8), nopython=True)
def _set_visibility(visibility_grid, i, j, value):
    visibility_grid[i][j] = value
    return


@jit(nb.b1(nb.i8, nb.i8, nb.f8, nb.f8, nb.f8,
           nb.f8, nb.i8, nb.i8, nb.i8, nb.f8), nopython=True)
def _outside_max_dist(viewpoint_row, viewpoint_col, west, ew_res, north,
                      ns_res, proj, row, col, max_distance):
    # Determine if the point at (row,col) is outside the maximum distance.

    if max_distance == INF:
        return False

    dist = _g_distance(_col_to_east(viewpoint_col + 0.5, west, ew_res),
                       _row_to_north(viewpoint_row + 0.5, north, ns_res),
                       _col_to_east(col + 0.5, west, ew_res),
                       _row_to_north(row + 0.5, north, ns_res),
                       proj)

    if max_distance < dist:
        return True

    return False


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


@jit(nb.b1(nb.f8), nopython=True)
def _is_null(value):
    # Check if a value is null.
    if value == NAN:
        return True
    return False


@jit(nb.f8(nb.i8, nb.i8, nb.i8, nb.i8, nb.i8,
           nb.i8, nb.i8, nb.i8[:, :]), nopython=True)
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
        if _is_null(elev1) or _is_null(elev2) or _is_null(elev3) \
                or _is_null(elev4):
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
           nb.f8, nb.f8, nb.f8, nb.f8, nb.f8, nb.i8), nopython=True)
def _calc_event_grad(row, col, elev, viewpoint_row, viewpoint_col,
                     viewpoint_elev, west, ew_res, north, ns_res, proj):
    # Calculate event gradient

    diff_elev = elev - viewpoint_elev

    if proj == PROJ_LL:
        dist = _g_distance(_col_to_east(col + 0.5, west, ew_res),
                           _row_to_north(row + 0.5, north, ns_res),
                           _col_to_east(viewpoint_col + 0.5, west, ew_res),
                           _row_to_north(viewpoint_row + 0.5, north, ns_res),
                           proj)

        distance_to_viewpoint = dist * dist

    else:
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
@jit(nb.types.Tuple((nb.f8, nb.f8))(nb.i8, nb.i8, nb.f8,
                                    nb.i8, nb.i8, nb.f8,
                                    nb.f8, nb.f8, nb.f8, nb.f8, nb.i8),
     nopython=True)
def _calc_dist_n_grad(status_node_row, status_node_col, elev,
                      viewpoint_row, viewpoint_col, viewpoint_elev,
                      west, ew_res, north, ns_res, proj):
    diff_elev = elev - viewpoint_elev

    if proj == PROJ_LL:
        dist = _g_distance(_col_to_east(status_node_col + 0.5, west, ew_res),
                           _row_to_north(status_node_row + 0.5, north, ns_res),
                           _col_to_east(viewpoint_col + 0.5, west, ew_res),
                           _row_to_north(viewpoint_row + 0.5, north, ns_res),
                           proj)

        distance_to_viewpoint = dist * dist
    else:
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
@jit(nb.void(nb.f8[:, :], nb.f8[:, :], nb.f8[:], nb.f8[:], nb.f8[:],
             nb.f8[:, :], nb.f8[:, :]), nopython=True)
def _init_event_list(event_list, raster, vp, v_op, g_hd, data,
                     visibility_grid):
    # Initialize and fill all the events for the map into event_list

    n_rows, n_cols = raster.shape
    inrast = np.empty(shape=(3, n_cols), dtype=np.int64)
    inrast.fill(NAN)

    # scan through the raster data
    # read first row
    inrast[2] = raster[0]

    # index of the event array: row, col, elev_0, elev_1, elev_2, ang, type
    e = np.zeros((7,), dtype=np.float64)
    e[E_ROW_ID] = -1
    e[E_COL_ID] = -1
    e[E_ELVEV_0] = NAN
    e[E_ELVEV_1] = NAN
    e[E_ELVEV_2] = NAN
    e[E_ANG_ID] = np.nan
    e[E_TYPE_ID] = np.nan

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
                inrast[2][j] = NAN

        # fill event list with events from this row
        for j in range(n_cols):
            e[E_ROW_ID] = i
            e[E_COL_ID] = j

            # read the elevation value into the event
            e[E_ELVEV_1] = inrast[1][j]

            # adjust for curvature
            e[E_ELVEV_1] = _adjust_curv(vp[VP_ROW_ID], vp[VP_COL_ID], i, j,
                                        e[E_ELVEV_1], v_op[VO_CURVE_ID],
                                        v_op[VO_ELLPS_A_ID], v_op[VO_REFR_ID],
                                        v_op[VO_REFR_COEF_ID],
                                        g_hd[GH_WEST_ID], g_hd[GH_EW_RES_ID],
                                        g_hd[GH_NORTH_ID], g_hd[GH_NS_RES_ID],
                                        g_hd[GH_PROJ_ID])

            # write it into the row of data going through the vp
            if i == vp[VP_ROW_ID]:
                data[0][j] = e[E_ELVEV_1]
                data[1][j] = e[E_ELVEV_1]
                data[2][j] = e[E_ELVEV_1]

            # set the vp, and don't insert it into eventlist
            if i == vp[VP_ROW_ID] and j == vp[VP_COL_ID]:

                # set_viewpoint_elev(vp, e.elev[1] + v_op.obsElev)
                vp[VP_ELEV_ID] = e[E_ELVEV_1] + v_op[VO_OBS_ELEV_ID]

                if v_op[VO_TARGET_ID] > 0:
                    vp[VP_TARGET_ID] = v_op[VO_TARGET_ID]
                else:
                    vp[VP_TARGET_ID] = 0.0

                _set_visibility(visibility_grid, i, j, 180)
                continue

            # if point is outside maxDist, do NOT include it as an event
            if _outside_max_dist(vp[VP_ROW_ID], vp[VP_COL_ID],
                                 g_hd[GH_WEST_ID], g_hd[GH_EW_RES_ID],
                                 g_hd[GH_NORTH_ID], g_hd[GH_NS_RES_ID],
                                 g_hd[GH_PROJ_ID],
                                 i, j, v_op[VO_MAX_DIST_ID]):
                continue

            # if it got here it is not the vp, not NODATA, and
            # within max distance from vp generate its 3 events
            # and insert them

            # get ENTER elevation
            e[E_TYPE_ID] = ENTERING_EVENT
            e[E_ELVEV_0] = _calc_event_elev(e[E_TYPE_ID], e[E_ROW_ID],
                                            e[E_COL_ID], n_rows, n_cols,
                                            vp[VP_ROW_ID], vp[VP_COL_ID],
                                            inrast)

            # adjust for curvature
            if v_op[VO_CURVE_ID]:
                ay, ax = _calc_event_pos(e[E_TYPE_ID], e[E_ROW_ID],
                                         e[E_COL_ID], vp[VP_ROW_ID],
                                         vp[VP_COL_ID])

                e[E_ELVEV_0] = _adjust_curv(vp[VP_ROW_ID], vp[VP_COL_ID],
                                            ay, ax, e[E_ELVEV_0],
                                            v_op[VO_CURVE_ID],
                                            v_op[VO_ELLPS_A_ID],
                                            v_op[VO_REFR_ID],
                                            v_op[VO_REFR_COEF_ID],
                                            g_hd[GH_WEST_ID],
                                            g_hd[GH_EW_RES_ID],
                                            g_hd[GH_NORTH_ID],
                                            g_hd[GH_NS_RES_ID],
                                            g_hd[GH_PROJ_ID])

            # get EXIT event
            e[E_TYPE_ID] = EXITING_EVENT
            e[E_ELVEV_2] = _calc_event_elev(e[E_TYPE_ID], e[E_ROW_ID],
                                            e[E_COL_ID], n_rows, n_cols,
                                            vp[VP_ROW_ID], vp[VP_COL_ID],
                                            inrast)

            # adjust for curvature
            if v_op[VO_CURVE_ID]:
                ay, ax = _calc_event_pos(e[E_TYPE_ID], e[E_ROW_ID],
                                         e[E_COL_ID], vp[VP_ROW_ID],
                                         vp[VP_COL_ID])

                e[E_ELVEV_2] = _adjust_curv(vp[VP_ROW_ID], vp[VP_COL_ID],
                                            ay, ax, e[E_ELVEV_2],
                                            v_op[VO_CURVE_ID],
                                            v_op[VO_ELLPS_A_ID],
                                            v_op[VO_REFR_ID],
                                            v_op[VO_REFR_COEF_ID],
                                            g_hd[GH_WEST_ID],
                                            g_hd[GH_EW_RES_ID],
                                            g_hd[GH_NORTH_ID],
                                            g_hd[GH_NS_RES_ID],
                                            g_hd[GH_PROJ_ID])

            # write adjusted elevation into the row of data
            # going through the vp
            if i == vp[VP_ROW_ID]:
                data[0][j] = e[E_ELVEV_0]
                data[1][j] = e[E_ELVEV_1]
                data[2][j] = e[E_ELVEV_2]

            # put event into event list
            e[E_TYPE_ID] = ENTERING_EVENT
            ay, ax = _calc_event_pos(e[E_TYPE_ID], e[E_ROW_ID],
                                     e[E_COL_ID], vp[VP_ROW_ID], vp[VP_COL_ID])
            e[E_ANG_ID] = _calculate_angle(ax, ay,
                                           vp[VP_COL_ID], vp[VP_ROW_ID])
            event_list[count_event] = e
            count_event += 1

            e[E_TYPE_ID] = CENTER_EVENT
            ay, ax = _calc_event_pos(e[E_TYPE_ID], e[E_ROW_ID],
                                     e[E_COL_ID], vp[VP_ROW_ID], vp[VP_COL_ID])
            e[E_ANG_ID] = _calculate_angle(ax, ay,
                                           vp[VP_COL_ID], vp[VP_ROW_ID])
            event_list[count_event] = e
            count_event += 1

            e[E_TYPE_ID] = EXITING_EVENT
            ay, ax = _calc_event_pos(e[E_TYPE_ID], e[E_ROW_ID], e[E_COL_ID],
                                     vp[VP_ROW_ID], vp[VP_COL_ID])
            e[E_ANG_ID] = _calculate_angle(ax, ay,
                                           vp[VP_COL_ID], vp[VP_ROW_ID])
            event_list[count_event] = e
            count_event += 1

    return


@jit(nb.i8(nb.f8[:, :]), nopython=True)
def _create_status_struct(tree):
    # Create and initialize the status struct.
    # return a Tree object with a dummy root.

    # dummy status node
    dummy_node_value = np.array([0.0, -1, -1, SMALLEST_GRAD, SMALLEST_GRAD,
                                SMALLEST_GRAD, 0.0, 0.0, 0.0, SMALLEST_GRAD])

    # node 0 is root
    root = 0
    _create_tree_node(tree, root, dummy_node_value, RB_BLACK)

    # last row is NIL
    _create_tree_node(tree, NIL_ID, dummy_node_value, RB_BLACK)
    _create_tree_node(tree, root, dummy_node_value, RB_BLACK)
    num_nodes = tree.shape[0]
    tree[NIL_ID][TN_LEFT_ID] = num_nodes
    tree[NIL_ID][TN_RIGHT_ID] = num_nodes
    tree[NIL_ID][TN_PARENT_ID] = num_nodes

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


@jit(nb.void(nb.f8[:], nb.i8, nb.i8), nopython=True)
def _init_status_node(status_node, row, col):
    status_node[S_ROW_ID] = row
    status_node[S_COL_ID] = col
    status_node[S_DIST2VP] = -1

    status_node[S_GRAD_0] = NAN
    status_node[S_GRAD_1] = NAN
    status_node[S_GRAD_2] = NAN

    status_node[S_ANG_0] = NAN
    status_node[S_ANG_1] = NAN
    status_node[S_ANG_2] = NAN

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
          'elevation = ', event[E_ELVEV_0], event[E_ELVEV_1], event[E_ELVEV_2],
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

@jit(nb.f8[:, :](nb.f8[:, :], nb.f8[:], nb.f8[:], nb.f8[:], nb.f8[:, :],
                 nb.f8[:, :], nb.f8[:, :]), nopython=True)
def _viewshed(raster, vp, v_op, g_hd, event_list, data, visibility_grid):
    n_rows, n_cols = raster.shape

    # for e in event_list:
    #     _print_event(e)

    # create the status structure
    # create 2d array of the RB-tree
    num_nodes = n_cols - int(vp[VP_COL_ID]) + n_cols * n_rows + 10
    status_struct = np.zeros((num_nodes, 12))
    root = _create_status_struct(status_struct)

    # idle row idx in the 2d data array of status_struct tree
    idle = np.zeros((num_nodes,), dtype=np.int64)
    for i in range(0, num_nodes - 1):
        idle[i] = num_nodes - i
    idle[0] = num_nodes - 2

    # Put cells that are initially on the sweepline into status structure
    for i in range(int(vp[VP_COL_ID]) + 1, n_cols):
        status_node = np.zeros((9,), dtype=np.float64)
        _init_status_node(status_node, vp[VP_ROW_ID], i)

        e = np.zeros((7,), dtype=np.float64)
        e[E_ROW_ID] = vp[VP_ROW_ID]
        e[E_COL_ID] = i
        e[E_ANG_ID] = np.nan
        e[E_TYPE_ID] = np.nan

        e[E_ELVEV_0] = data[0][i]
        e[E_ELVEV_1] = data[1][i]
        e[E_ELVEV_2] = data[2][i]

        if (not _is_null(data[1][i])) and \
                (not _outside_max_dist(vp[VP_ROW_ID], vp[VP_COL_ID],
                                       g_hd[GH_WEST_ID], g_hd[GH_EW_RES_ID],
                                       g_hd[GH_NORTH_ID], g_hd[GH_NS_RES_ID],
                                       g_hd[GH_PROJ_ID], status_node[S_ROW_ID],
                                       status_node[S_COL_ID],
                                       v_op[VO_MAX_DIST_ID])):
            # calculate Distance to VP and Gradient,
            # store them into status_node
            # need either 3 elevation values or
            # 3 gradients calculated from 3 elevation values
            # need also 3 angs

            e[E_TYPE_ID] = ENTERING_EVENT
            ay, ax = _calc_event_pos(e[E_TYPE_ID], e[E_ROW_ID], e[E_COL_ID],
                                     vp[VP_ROW_ID], vp[VP_COL_ID])
            status_node[S_ANG_0] = _calculate_angle(ax, ay, vp[VP_COL_ID],
                                                    vp[VP_ROW_ID])
            status_node[S_GRAD_0] = _calc_event_grad(ay, ax, e[E_ELVEV_0],
                                                     vp[VP_ROW_ID],
                                                     vp[VP_COL_ID],
                                                     vp[VP_ELEV_ID],
                                                     g_hd[GH_WEST_ID],
                                                     g_hd[GH_EW_RES_ID],
                                                     g_hd[GH_NORTH_ID],
                                                     g_hd[GH_NS_RES_ID],
                                                     g_hd[GH_PROJ_ID])

            e[E_TYPE_ID] = CENTER_EVENT
            ay, ax = _calc_event_pos(e[E_TYPE_ID], e[E_ROW_ID], e[E_COL_ID],
                                     vp[VP_ROW_ID], vp[VP_COL_ID])
            status_node[S_ANG_1] = _calculate_angle(ax, ay, vp[VP_COL_ID],
                                                    vp[VP_ROW_ID])
            status_node[S_DIST2VP], status_node[S_GRAD_1] = \
                _calc_dist_n_grad(status_node[S_ROW_ID], status_node[S_COL_ID],
                                  e[E_ELVEV_1], vp[VP_ROW_ID], vp[VP_COL_ID],
                                  vp[VP_ELEV_ID], g_hd[GH_WEST_ID],
                                  g_hd[GH_EW_RES_ID], g_hd[GH_NORTH_ID],
                                  g_hd[GH_NS_RES_ID], g_hd[GH_PROJ_ID])

            e[E_TYPE_ID] = EXITING_EVENT
            ay, ax = _calc_event_pos(e[E_TYPE_ID], e[E_ROW_ID], e[E_COL_ID],
                                     vp[VP_ROW_ID], vp[VP_COL_ID])
            status_node[S_ANG_2] = _calculate_angle(ax, ay, vp[VP_COL_ID],
                                                    vp[VP_ROW_ID])
            status_node[S_GRAD_2] = _calc_event_grad(ay, ax, e[E_ELVEV_2],
                                                     vp[VP_ROW_ID],
                                                     vp[VP_COL_ID],
                                                     vp[VP_ELEV_ID],
                                                     g_hd[GH_WEST_ID],
                                                     g_hd[GH_EW_RES_ID],
                                                     g_hd[GH_NORTH_ID],
                                                     g_hd[GH_NS_RES_ID],
                                                     g_hd[GH_PROJ_ID])

            assert status_node[S_ANG_1] == 0

            if status_node[S_ANG_0] > status_node[S_ANG_1]:
                status_node[S_ANG_0] -= 2 * PI

            # insert sn into the status structure
            id = _pop(idle)
            root = _insert_into_tree(status_struct, root, id, status_node)

    # sweep the event_list

    # number of visible cells
    nvis = 0
    nevents = len(event_list)

    for i in range(nevents):
        # get out one event at a time and process it according to its type
        e = event_list[i]

        # status_node = StatusNode(row=e[E_ROW_ID], col=e[E_COL_ID])
        status_node = np.zeros((9,), dtype=np.float64)
        _init_status_node(status_node, e[E_ROW_ID], e[E_COL_ID])

        # calculate Distance to VP and Gradient
        status_node[S_DIST2VP], status_node[S_GRAD_1] = \
            _calc_dist_n_grad(status_node[S_ROW_ID], status_node[S_COL_ID],
                              e[E_ELVEV_1] + vp[VP_TARGET_ID],
                              vp[VP_ROW_ID], vp[VP_COL_ID], vp[VP_ELEV_ID],
                              g_hd[GH_WEST_ID], g_hd[GH_EW_RES_ID],
                              g_hd[GH_NORTH_ID], g_hd[GH_NS_RES_ID],
                              g_hd[GH_PROJ_ID])

        etype = e[E_TYPE_ID]
        if etype == ENTERING_EVENT:
            # insert node into structure

            #  need either 3 elevation values or
            # 	     * 3 gradients calculated from 3 elevation values */
            # 	    /* need also 3 angs */
            ay, ax = _calc_event_pos(e[E_TYPE_ID], e[E_ROW_ID], e[E_COL_ID],
                                     vp[VP_ROW_ID], vp[VP_COL_ID])
            status_node[S_ANG_0] = e[E_ANG_ID]
            status_node[S_GRAD_0] = _calc_event_grad(ay, ax, e[E_ELVEV_0],
                                                     vp[VP_ROW_ID],
                                                     vp[VP_COL_ID],
                                                     vp[VP_ELEV_ID],
                                                     g_hd[GH_WEST_ID],
                                                     g_hd[GH_EW_RES_ID],
                                                     g_hd[GH_NORTH_ID],
                                                     g_hd[GH_NS_RES_ID],
                                                     g_hd[GH_PROJ_ID])

            e[E_TYPE_ID] = CENTER_EVENT
            ay, ax = _calc_event_pos(e[E_TYPE_ID], e[E_ROW_ID], e[E_COL_ID],
                                     vp[VP_ROW_ID], vp[VP_COL_ID])
            status_node[S_ANG_1] = _calculate_angle(ax, ay, vp[VP_COL_ID],
                                                    vp[VP_ROW_ID])
            status_node[S_DIST2VP], status_node[S_GRAD_1] = \
                _calc_dist_n_grad(status_node[S_ROW_ID], status_node[S_COL_ID],
                                  e[E_ELVEV_1], vp[VP_ROW_ID], vp[VP_COL_ID],
                                  vp[VP_ELEV_ID], g_hd[GH_WEST_ID],
                                  g_hd[GH_EW_RES_ID], g_hd[GH_NORTH_ID],
                                  g_hd[GH_NS_RES_ID], g_hd[GH_PROJ_ID])

            e[E_TYPE_ID] = EXITING_EVENT
            ay, ax = _calc_event_pos(e[E_TYPE_ID], e[E_ROW_ID], e[E_COL_ID],
                                     vp[VP_ROW_ID], vp[VP_COL_ID])
            status_node[S_ANG_2] = _calculate_angle(ax, ay, vp[VP_COL_ID],
                                                    vp[VP_ROW_ID])
            status_node[S_GRAD_2] = _calc_event_grad(ay, ax, e[E_ELVEV_2],
                                                     vp[VP_ROW_ID],
                                                     vp[VP_COL_ID],
                                                     vp[VP_ELEV_ID],
                                                     g_hd[GH_WEST_ID],
                                                     g_hd[GH_EW_RES_ID],
                                                     g_hd[GH_NORTH_ID],
                                                     g_hd[GH_NS_RES_ID],
                                                     g_hd[GH_PROJ_ID])

            e[E_TYPE_ID] = ENTERING_EVENT

            if e[E_ANG_ID] < PI:
                if status_node[S_ANG_0] > status_node[S_ANG_1]:
                    status_node[S_ANG_0] -= 2 * PI
            else:
                if status_node[S_ANG_0] > status_node[S_ANG_1]:
                    status_node[S_ANG_1] += 2 * PI
                    status_node[S_ANG_2] += 2 * PI

            id = _pop(idle)
            root = _insert_into_tree(status_struct, root, id, status_node)

        elif etype == EXITING_EVENT:
            # delete node out of status structure
            root, deleted = _delete_from_tree(status_struct, root,
                                              status_node[S_DIST2VP])
            _push(idle, deleted)

        elif etype == CENTER_EVENT:
            # calculate visibility
            # consider current ang and gradient
            max = _max_grad_in_status_struct(status_struct, root,
                                             status_node[S_DIST2VP],
                                             e[E_ANG_ID],
                                             status_node[S_GRAD_1])

            # the point is visible: store its vertical ang
            if max <= status_node[S_GRAD_1]:
                vert_ang = _get_vertical_ang(vp[VP_ELEV_ID],
                                             status_node[S_DIST2VP],
                                             e[E_ELVEV_1] + vp[VP_TARGET_ID])

                _set_visibility(visibility_grid, status_node[S_ROW_ID],
                                status_node[S_COL_ID], vert_ang)

                assert vert_ang >= 0
                # when you write the visibility grid you assume that
                # 		   visible values are positive

                nvis += 1

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

    # TODO: Remove these in the future ---
    do_curve = DO_CURVE
    do_refr = DO_REFR
    max_distance = INF
    proj = PROJ_NONE
    # ------------------------------------

    # viewpoint = ViewPoint(row=y_view, col=x_view)
    viewpoint = np.zeros((4,), dtype=np.float64)
    viewpoint[VP_ROW_ID] = y_view
    viewpoint[VP_COL_ID] = x_view

    # view_options = ViewOptions(obs_elev=observer_elev, tgt_elev=target_elev,
    #                            max_dist=max_distance,
    #                            do_curv=do_curve, do_refr=do_refr)
    view_options = np.zeros((7,), dtype=np.float64)
    view_options[VO_OBS_ELEV_ID] = observer_elev
    view_options[VO_TARGET_ID] = target_elev
    view_options[VO_MAX_DIST_ID] = max_distance
    view_options[VO_CURVE_ID] = do_curve
    view_options[VO_REFR_ID] = do_refr
    view_options[VO_ELLPS_A_ID] = ELLPS_A
    view_options[VO_REFR_COEF_ID] = REFR_COEF

    # int getgrdhead(FILE * fd, struct Cell_head *cellhd)
    grid_header = np.zeros((7,), dtype=np.float64)
    grid_header[GH_PROJ_ID] = proj
    grid_header[GH_EW_RES_ID] = (x_range[1] - x_range[0]) / (width - 1)
    grid_header[GH_NS_RES_ID] = (y_range[1] - y_range[0]) / (height - 1)
    grid_header[GH_NORTH_ID] = y_range[1] + grid_header[GH_NS_RES_ID] / 2.0
    grid_header[GH_SOUTH_ID] = y_range[0] - grid_header[GH_NS_RES_ID] / 2.0
    grid_header[GH_EAST_ID] = x_range[1] + grid_header[GH_EW_RES_ID] / 2.0
    grid_header[GH_WEST_ID] = x_range[0] - grid_header[GH_EW_RES_ID] / 2.0

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
                     vp=viewpoint, v_op=view_options, g_hd=grid_header,
                     data=data, visibility_grid=visibility_grid)

    # sort the events radially by ang
    event_list = event_list[np.lexsort((event_list[:, E_TYPE_ID],
                                        event_list[:, E_ANG_ID]))]

    viewshed_img = _viewshed(raster.values, viewpoint, view_options,
                             grid_header, event_list, data, visibility_grid)

    visibility = xarray.DataArray(viewshed_img,
                                  coords=raster.coords,
                                  attrs=raster.attrs,
                                  dims=raster.dims)
    return visibility
