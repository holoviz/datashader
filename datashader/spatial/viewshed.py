import xarray
import numpy as np
import math
from math import atan, sqrt, fabs
import numba as nb
from numba import jit


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
SMALLEST_GRAD = -9999999999999999999999.0

PROJ_LL = 0
PROJ_NONE = -1

PI = math.pi

NAN = -9999999999999999


class TreeValue:

    # Implementation of value in a tree node.
    # A tree value represents for an entry (or cell or pixel) in the raster.
    # There are 3 events occur in each entry:
    #     ENTERING_EVENT, CENTER_EVENT, EXITING_EVENT
    #
    # Attributes:
    #     key: float, distance from the cell corresponding to the tree node to
    #             the vp
    #     gradient: 1d array of 3 elements, gradients wrt vp of ENTERING,
    #                 CENTER and EXITING events in the corresponding cell.
    #     ang: 1d array of 3 elements, angs wrt vp of ENTERING, CENTER
    #                 and EXITING events in the corresponding cell.
    #     max_grad: float, max_grad within the distance from
    #                     the corresponding cell to vp.

    __slots__ = ('key', 'grad', 'ang', 'max_grad')

    def __init__(self, key, gradient=[], ang=[],
                 max_grad=SMALLEST_GRAD):
        self.key = key

        if not len(gradient):
            gradient = np.empty(shape=(3,))
            gradient.fill(NAN)
            self.grad = gradient
        else:
            self.grad = np.array(gradient)

        if not len(ang):
            ang = np.empty(shape=(3,))
            ang.fill(NAN)
            self.ang = ang
        else:
            self.ang = np.array(ang)

        self.max_grad = max_grad

    # find the min value in the given tree value
    def find_value_min_value(self):
        if self.grad[0] < self.grad[1]:
            if self.grad[0] < self.grad[2]:
                return self.grad[0]
            else:
                return self.grad[2]
        else:
            if self.grad[1] < self.grad[2]:
                return self.grad[1]
            else:
                return self.grad[2]

        return self.grad[0]


class TreeNode:

    # Implementation of tree node.
    # Attributes:
    #     value: TreeValue, value of the node.
    #     color: int, RB_RED or RB_BLACK
    #     left: TreeNode, the left node of the this tree node
    #     right: TreeNode, the right node of the this tree node
    #     parent: TreeNode, the parent node of the this tree node

    __slots__ = ('val', 'color', 'left', 'right', 'parent')

    def __init__(self, tree_val, color, left=None, right=None, parent=None):
        self.val = tree_val
        self.color = color
        self.left = left
        self.right = right
        self.parent = parent

    # for debug purpose, traverse through the tree
    def __iter__(self):

        if self.left != NIL:
            for l in self.left.__iter__():
                yield l

        yield self.val.key

        if self.right != NIL:
            for r in self.right.__iter__():
                yield r


NIL_VALUE = TreeValue(key=0,
                      gradient=[SMALLEST_GRAD, SMALLEST_GRAD, SMALLEST_GRAD],
                      ang=[0, 0, 0], max_grad=SMALLEST_GRAD)

NIL = TreeNode(tree_val=NIL_VALUE, color=RB_BLACK)


# function used by treeSuccessor
def _tree_minimum(x):
    while x.left != NIL:
        x = x.left
    return x


# function used by deletion
def _tree_successor(x):
    # Find the highest successor of a node in the tree

    if x.right != NIL:
        return _tree_minimum(x.right)

    y = x.parent
    while y != NIL and x == y.right:
        x = y
        if y.parent == NIL:
            return y
        y = y.parent
    return y


def _find_max_value(node):
    # Find the max value in the given tree.
    if node is None:
        return SMALLEST_GRAD

    return node.val.max_grad


class Tree:
    # Implementation of red black tree.
    # Attribute:
    #     root: TreeNode, root of the tree.

    __slots__ = ('root')

    def __init__(self, root=None):
        self.root = root

    # for debug purpose, traverse through the tree
    def __iter__(self):

        if not self.root:
            yield list()

        else:
            for r in self.root.__iter__():
                yield r

    def _left_rotate(self, x):
        # A utility function to left rotate subtree rooted with a node.

        y = x.right

        # fix x
        if x.left.val.max_grad > y.left.val.max_grad:
            tmp_max = x.left.val.max_grad
        else:
            tmp_max = y.left.val.max_grad

        if tmp_max > x.val.find_value_min_value():
            x.val.max_grad = tmp_max
        else:
            x.val.max_grad = x.val.find_value_min_value()

        # fix y
        if x.val.max_grad > y.right.val.max_grad:
            tmp_max = x.val.max_grad
        else:
            tmp_max = y.right.val.max_grad

        if tmp_max > y.val.find_value_min_value():
            y.val.max_grad = tmp_max
        else:
            y.val.max_grad = y.val.find_value_min_value()

        # left rotation
        # see pseudo code on page 278 CLRS

        # turn y's left subtree into x's right subtree
        x.right = y.left
        y.left.parent = x
        # link x's parent to y
        y.parent = x.parent

        if x.parent == NIL:
            self.root = y
        else:
            if x == x.parent.left:
                x.parent.left = y
            else:
                x.parent.right = y

        y.left = x
        x.parent = y
        return

    def _right_rotate(self, y):
        # A utility function to right rotate subtree rooted with a node.

        x = y.left

        # fix y
        if x.right.val.max_grad > y.right.val.max_grad:
            tmp_max = x.right.val.max_grad
        else:
            tmp_max = y.right.val.max_grad

        if tmp_max > y.val.find_value_min_value():
            y.val.max_grad = tmp_max
        else:
            y.val.max_grad = y.val.find_value_min_value()

        # fix x
        if x.left.val.max_grad > y.val.max_grad:
            tmp_max = x.left.val.max_grad
        else:
            tmp_max = y.val.max_grad

        if tmp_max > x.val.find_value_min_value():
            x.val.max_grad = tmp_max
        else:
            x.val.max_grad = x.val.find_value_min_value()

        # rotation
        y.left = x.right
        x.right.parent = y

        x.parent = y.parent

        if y.parent == NIL:
            self.root = x
        else:
            if y.parent.left == y:
                y.parent.left = x
            else:
                y.parent.right = x

        x.right = y
        y.parent = x
        return

    def _rb_insert_fixup(self, z):
        # Fix red-black tree after insertion. This may change the root pointer.

        # see pseudocode on page 281 in CLRS
        while z.parent.color == RB_RED:
            if z.parent == z.parent.parent.left:
                y = z.parent.parent.right
                if y.color == RB_RED:
                    # case 1
                    z.parent.color = RB_BLACK
                    y.color = RB_BLACK
                    z.parent.parent.color = RB_RED
                    z = z.parent.parent
                else:
                    if z == z.parent.right:
                        # case 2
                        z = z.parent
                        # convert case 2 to case 3
                        self._left_rotate(z)
                    # case 3
                    z.parent.color = RB_BLACK
                    z.parent.parent.color = RB_RED
                    self._right_rotate(z.parent.parent)

            else:
                # (z->parent == z->parent->parent->right)
                y = z.parent.parent.left
                if y.color == RB_RED:
                    # case 1
                    z.parent.color = RB_BLACK
                    y.color = RB_BLACK
                    z.parent.parent.color = RB_RED
                    z = z.parent.parent
                else:
                    if z == z.parent.left:
                        # case 2
                        z = z.parent
                        # convert case 2 to case 3
                        self._right_rotate(z)
                    # case 3
                    z.parent.color = RB_BLACK
                    z.parent.parent.color = RB_RED
                    self._left_rotate(z.parent.parent)

        self.root.color = RB_BLACK
        return

    def insert_into_tree(self, value):
        # Create node and insert it into the tree

        cur_node = self.root

        if _compare(value.key, cur_node.val.key) == -1:
            next_node = cur_node.left
        else:
            next_node = cur_node.right

        while next_node != NIL:
            cur_node = next_node
            if _compare(value.key, cur_node.val.key) == -1:
                next_node = cur_node.left
            else:
                next_node = cur_node.right

        # create a new node
        #   //and place it at the right place
        #   //created node is RED by default */
        next_node = _create_tree_node(value)

        next_node.parent = cur_node

        if _compare(value.key, cur_node.val.key) == -1:
            cur_node.left = next_node
        else:
            cur_node.right = next_node

        inserted = next_node

        # update augmented maxGradient
        next_node.val.max_grad = next_node.val.find_value_min_value()
        while next_node.parent != NIL:
            if next_node.parent.val.max_grad < next_node.val.max_grad:
                next_node.parent.val.max_grad = next_node.val.max_grad

            if next_node.parent.val.max_grad > next_node.val.max_grad:
                break

            next_node = next_node.parent

        # fix rb tree after insertion
        self._rb_insert_fixup(inserted)
        return

    def _search_for_node(self, key):
        # Search for a node with a given key.

        cur_node = self.root
        while cur_node != NIL and \
                _compare(key, cur_node.val.key) != 0:
            if _compare(key, cur_node.val.key) == -1:
                cur_node = cur_node.left
            else:
                cur_node = cur_node.right

        return cur_node

    # The following is designed for viewshed's algorithm
    def find_max_value_within_key(self, max_key, ang, gradient):
        key_node = self._search_for_node(max_key)
        if key_node == NIL:
            # there is no point in the structure with key < maxKey */
            return SMALLEST_GRAD

        cur_node = key_node
        max = SMALLEST_GRAD

        while cur_node.parent != NIL:
            if cur_node == cur_node.parent.right:
                # its the right node of its parent
                tmp_max = _find_max_value(cur_node.parent.left)
                if tmp_max > max:
                    max = tmp_max
                if cur_node.parent.val.find_value_min_value() > max:
                    max = cur_node.parent.val.find_value_min_value()
            cur_node = cur_node.parent

        if max > gradient:
            return max

        # traverse all nodes with smaller distance
        max = SMALLEST_GRAD
        cur_node = key_node
        while cur_node != NIL:
            check_me = False
            if cur_node.val.ang[0] <= ang \
                    <= cur_node.val.ang[2]:
                check_me = True
            if (not check_me) and cur_node.val.key > 0:
                print('Angles outside angle')

            if cur_node.val.key > max_key:
                raise ValueError("current dist too large ")

            if check_me and cur_node != key_node:

                if ang < cur_node.val.ang[1]:
                    cur_grad = cur_node.val.grad[1] \
                        + (cur_node.val.grad[0] - cur_node.val.grad[1])\
                        * (cur_node.val.ang[1] - ang) \
                        / (cur_node.val.ang[1] - cur_node.val.ang[0])

                elif ang > cur_node.val.ang[1]:
                    cur_grad = cur_node.val.grad[1] \
                        + (cur_node.val.grad[2] - cur_node.val.grad[1])\
                        * (ang - cur_node.val.ang[1]) \
                        / (cur_node.val.ang[2] - cur_node.val.ang[1])
                else:
                    cur_grad = cur_node.val.grad[1]

                if cur_grad > max:
                    max = cur_grad

                if max > gradient:
                    return max

            # get next smaller key
            if cur_node.left != NIL:
                cur_node = cur_node.left
                while cur_node.right != NIL:
                    cur_node = cur_node.right
            else:
                # at smallest item in this branch, go back up
                last_node = cur_node
                cur_node = cur_node.parent
                while cur_node != NIL and last_node == cur_node.left:
                    last_node = cur_node
                    cur_node = cur_node.parent

        return max

    def _rb_delete_fixup(self, x):
        # Fix the red-black tree after deletion.
        # This may change the root pointer.

        while x != self.root and x.color == RB_BLACK:
            if x == x.parent.left:
                w = x.parent.right
                if w.color == RB_RED:
                    w.color = RB_BLACK
                    x.parent.color = RB_RED
                    self._left_rotate(x.parent)
                    w = x.parent.right

                if w == NIL:
                    x = x.parent
                    continue

                if w.left.color == RB_BLACK and w.right.color == RB_BLACK:
                    w.color = RB_RED
                    x = x.parent
                else:
                    if w.right.color == RB_BLACK:
                        w.left.color = RB_BLACK
                        w.color = RB_RED
                        self._right_rotate(w)
                        w = x.parent.right
                    w.color = x.parent.color
                    x.parent.color = RB_BLACK
                    w.right.color = RB_BLACK
                    self._left_rotate(x.parent)
                    x = self.root
            else:
                # x == x.parent.right
                w = x.parent.left
                if w.color == RB_RED:
                    w.color = RB_BLACK
                    x.parent.color = RB_RED
                    self._right_rotate(x.parent)
                    w = x.parent.left

                if w == NIL:
                    x = x.parent
                    continue

                if w.right.color == RB_BLACK and w.left.color == RB_BLACK:
                    w.color = RB_RED
                    x = x.parent
                else:
                    if w.left.color == RB_BLACK:
                        w.right.color = RB_BLACK
                        w.color = RB_RED
                        self._left_rotate(w)
                        w = x.parent.left
                    w.color = x.parent.color
                    x.parent.color = RB_BLACK
                    w.left.color = RB_BLACK
                    self._right_rotate(x.parent)
                    x = self.root

        x.color = RB_BLACK
        return

    def _delete_from_tree(self, key):
        # Delete the node out of the tree. This may change the root pointer.

        z = self._search_for_node(key)

        if z == NIL:
            # node to delete is not found
            raise ValueError("node not found")

        # 1-3
        if z.left == NIL or z.right == NIL:
            y = z
        else:
            y = _tree_successor(z)

        if y == NIL:
            raise ValueError("successor not found")

        # 4-6
        if y.left != NIL:
            x = y.left
        else:
            x = y.right

        # 7
        x.parent = y.parent

        # 8-12
        if y.parent == NIL:
            self.root = x
            # augmentation to be fixed
            to_fix = self.root
        else:
            if y == y.parent.left:
                y.parent.left = x
            else:
                y.parent.right = x
            # augmentation to be fixed
            to_fix = y.parent

        # fix augmentation for removing y
        cur_node = y

        while cur_node.parent != NIL:
            if cur_node.parent.val.max_grad == y.val.find_value_min_value():
                left = _find_max_value(cur_node.parent.left)
                right = _find_max_value(cur_node.parent.right)

                if left > right:
                    cur_node.parent.val.max_grad = left
                else:
                    cur_node.parent.val.max_grad = right

                if cur_node.parent.val.find_value_min_value() > \
                        cur_node.parent.val.max_grad:
                    cur_node.parent.val.max_grad = \
                        cur_node.parent.val.find_value_min_value()

            else:
                break

            cur_node = cur_node.parent

        # fix augmentation for x
        if to_fix.left.val.max_grad > to_fix.right.val.max_grad:
            tmp_max = to_fix.left.val.max_grad
        else:
            tmp_max = to_fix.right.val.max_grad

        if tmp_max > to_fix.val.find_value_min_value():
            to_fix.val.max_grad = tmp_max
        else:
            to_fix.val.max_grad = to_fix.val.find_value_min_value()

        # 13-15
        if y != NIL and y != z:
            z_gradient = z.val.find_value_min_value()

            z.val.key = y.val.key
            z.val.grad[0] = y.val.grad[0]
            z.val.grad[1] = y.val.grad[1]
            z.val.grad[2] = y.val.grad[2]
            z.val.ang[0] = y.val.ang[0]
            z.val.ang[1] = y.val.ang[1]
            z.val.ang[2] = y.val.ang[2]

            to_fix = z
            # fix augmentation
            if to_fix.left.val.max_grad > to_fix.right.val.max_grad:
                tmp_max = to_fix.left.val.max_grad
            else:
                tmp_max = to_fix.right.val.max_grad

            if tmp_max > to_fix.val.find_value_min_value():
                to_fix.val.max_grad = tmp_max
            else:
                to_fix.val.max_grad = to_fix.val.find_value_min_value()

            while z.parent != NIL:
                if z.parent.val.max_grad == z_gradient:
                    if z.parent.val.find_value_min_value() != z_gradient and \
                        not (z.parent.left.val.max_grad == z_gradient) and \
                       x.parent.right.val.max_grad == z_gradient:

                        left = _find_max_value(z.parent.left)
                        right = _find_max_value(z.parent.right)

                        if left > right:
                            z.parent.val.max_grad = left
                        else:
                            z.parent.val.max_grad = right

                        if z.parent.val.find_value_min_value() > \
                                z.parent.val.max_grad:
                            z.parent.val.max_grad = \
                                z.parent.val.find_value_min_value()

                else:
                    if z.val.max_grad > z.parent.val.max_grad:
                        z.parent.val.max_grad = z.val.max_grad

                z = z.parent

        # 16-17
        if y.color == RB_BLACK and x != NIL:
            self._rb_delete_fixup(x)

        # 18
        return


class StatusNode:
    # Implementation of status of sweeping process.
    # Attributes:
    #      row, col: int, row and col to determine the position of the cell
    #      distance_to_viewpoint: float, elevation of cell
    #      gradient: 1d array of 3 float elements, ENTER, CENTER,
    #                 EXIT gradients of the Line of Sight
    #      ang:   1d array of 3 float elements, ENTER, CENTER, EXIT angles of
    #                 the Line of Sight

    __slots__ = ('row', 'col', 'dist_to_viewpoint', 'grad', 'ang')

    def __init__(self, row, col, dist_to_vp=-1, gradient=[], angle=[]):
        # row and col to determine the position of the cell
        self.row = row
        self.col = col

        # elevation of cell
        self.dist_to_viewpoint = dist_to_vp

        # ENTER, CENTER, EXIT gradients of the Line of Sight
        if not len(gradient):
            gradient = np.empty(shape=(3,))
            gradient.fill(NAN)
            self.grad = gradient
        else:
            self.grad = np.array(gradient)

        # ENTER, CENTER, EXIT angs of the Line of Sight
        if not len(angle):
            ang = np.empty(shape=(3,))
            ang.fill(NAN)
            self.ang = ang
        else:
            self.ang = np.array(angle)

    def _print_status_node(self):
        print(self.row, self.col, self.dist_to_viewpoint,
              self.grad, self.ang)


def _insert_into_status_struct(status_node, tree):
    # Create a TreeValue object that get information from the input status_node
    # Insert the node with the created value into the tree

    tv = TreeValue(key=status_node.dist_to_viewpoint,
                   gradient=status_node.grad,
                   ang=status_node.ang,
                   max_grad=SMALLEST_GRAD)
    tree.insert_into_tree(value=tv)
    return


def _max_grad_in_status_struct(tree, distance, angle, gradient):
    # Find the node with max Gradient within the distance (from vp)
    # Note: if there is nothing in the status structure,
    #         it means this cell is VISIBLE

    if not tree.root:
        return SMALLEST_GRAD

    # it is also possible that the status structure is not empty, but
    # there are no events with key < dist ---in this case it returns
    # SMALLEST_GRAD;

    # find max within the max key

    return tree.find_max_value_within_key(distance, angle, gradient)


class GridHeader:
    # Implementation of grid header for the visibility grid that
    #     returned by _viewshed().
    # Attributes:
    #     proj: int, there can be no proj or
    #                     lat-lon proj type applied.
    #     cols: int, number of columns in the grid
    #     rows: int, number of rows in the grid
    #     ew_res: float, east-west resolution of the grid
    #     ns_res: float, north-south resolution of the grid
    #     north: float
    #     south: float
    #     east: float
    #     west: float

    __slots__ = ('proj', 'cols', 'rows', 'ew_res', 'ns_res',
                 'north', 'south', 'east', 'west')

    def __init__(self, width, height, x_range, y_range, proj=PROJ_NONE):

        # int getgrdhead(FILE * fd, struct Cell_head *cellhd)

        self.proj = proj

        self.cols = width
        self.rows = height

        self.ew_res = (x_range[1] - x_range[0]) / (self.cols - 1)
        self.ns_res = (y_range[1] - y_range[0]) / (self.rows - 1)

        self.north = y_range[1] + self.ns_res / 2.0
        self.south = y_range[0] - self.ns_res / 2.0
        self.east = x_range[1] + self.ew_res / 2.0
        self.west = x_range[0] - self.ew_res / 2.0


class ViewPoint:
    # Implementation of vp.
    # Attributes:
    #     row, col: int, coordinate of the vp
    #     elevation: float, elevation of the observer at the vp
    #     target_offset: float, target offset of the observer at the vp

    __slots__ = ('row', 'col', 'elev', 'target_offset')

    def __init__(self, row, col, elevation=0, target_offset=0):
        self.row = row
        self.col = col
        self.elev = elevation
        self.target_offset = target_offset


class ViewOptions:
    # Implementaion of view options.
    # Attributes:
    #     obs_elev: float, observer elevation above the terrain
    #     tgt_elev: float, target elevation offset above the terrain
    #     max_distance: float, points that are farther than this distance from
    #                     the vp are not visible
    #     do_curv: boolean, determines if the curvature of the earth should be
    #                     considered when calculating
    #     ellps_a: float, the parameter of the ellipsoid
    #     do_refr: boolean, determines if atmospheric refraction should be
    #                 considered when calculating
    #     refr_coef: float, atmospheric refraction coefficient

    __slots__ = ('obs_elev', 'tgt_elev', 'max_distance', 'do_curv', 'ellps_a',
                 'do_refr', 'refr_coef')

    def __init__(self, obs_elev=OBS_ELEV, tgt_elev=TARGET_ELEV, max_dist=INF,
                 do_curv=DO_CURVE, ellps_a=ELLPS_A,
                 do_refr=DO_REFR, refr_coef=REFR_COEF):

        # observer elevation above the terrain
        self.obs_elev = obs_elev

        # target elevation offset above the terrain
        self.tgt_elev = tgt_elev

        # points that are farther than this distance from the vp
        # are not visible
        self.max_distance = max_dist

        # determines if the curvature of the earth should be considered
        # when calculating
        self.do_curv = do_curv

        # the parameter of the ellipsoid
        self.ellps_a = ellps_a

        # determines if atmospheric refraction should be considered
        # when calculating
        self.do_refr = do_refr
        self.refr_coef = refr_coef


class Event:

    # Implementation of event.
    # There are 3 events in a cell: ENTERING_EVENT, CENTER_EVENT, EXITING_EVENT
    # Attributes:
    #     row, col: int, coordinate of cell that the CENTER event occurs
    #     elev: 1d array of 3 float elements, elevation of ENTERING_EVENT,
    #                 CENTER_EVENT, EXITING_EVENT
    #     ang: float
    #     event: int, ENTERING_EVENT, CENTER_EVENT, EXITING_EVENT

    __slots__ = ('row', 'col', 'elev', 'ang', 'type')

    def __init__(self, row=-1, col=-1, elev=[], angle=None, event_type=None):
        self.row = row
        self.col = col

        if not len(elev):
            elev = np.empty(shape=(3,))
            elev.fill(NAN)
            self.elev = elev
        else:
            self.elev = np.array(elev)

        self.ang = angle
        self.type = event_type

    # for debug purpose
    def _print_event(self):
        print('event_type = ', self.type,
              'row = ', self.row,
              'col = ', self.col,
              'elevation = ', self.elev,
              'ang = ', self.ang)


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
@jit(nb.f8(nb.i8, nb.i8, nb.i8, nb.i8, nb.f8, nb.b1, nb.f8, nb.b1,
           nb.f8, nb.f8, nb.f8, nb.f8, nb.f8, nb.i8), nopython=True)
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


@jit(nb.b1(nb.i8, nb.i8,
           nb.f8, nb.f8, nb.f8, nb.f8, nb.i8,
           nb.i8, nb.i8, nb.f8), nopython=True)
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
        return math.pi / 2

    if viewpoint_x == event_x and viewpoint_y < event_y:
        # between 3rd and 4th quadrant
        return math.pi * 3.0 / 2.0

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
        return math.pi - ang

    if viewpoint_x > event_x and viewpoint_y == event_y:
        # between 1st and 3rd quadrant
        return math.pi

    if viewpoint_x > event_x and viewpoint_y < event_y:
        # 3rd quadrant
        return math.pi + ang

    if viewpoint_x < event_x and viewpoint_y < event_y:
        # 4th quadrant
        return math.pi * 2.0 - ang

    assert event_x == viewpoint_x and event_y == viewpoint_y
    return 0


@jit(nb.f8(nb.i8, nb.i8, nb.f8,
           nb.i8, nb.i8, nb.f8,
           nb.f8, nb.f8, nb.f8, nb.f8, nb.i8), nopython=True)
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

    if diff_elev == 0:
        gradient = 0
        return distance_to_viewpoint, gradient

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

def _init_event_list(event_list, raster, vp, v_op, g_hd, data,
                     visibility_grid):
    # Initialize and fill all the events for the map into event_list

    n_rows, n_cols = raster.shape
    inrast = np.empty(shape=(3, n_cols), dtype=np.int64)
    inrast.fill(NAN)

    # scan through the raster data
    # read first row
    inrast[2] = raster[0]

    e = Event()
    e.ang = -1

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
            e.row = i
            e.col = j

            # read the elevation value into the event
            e.elev[1] = inrast[1][j]

            # adjust for curvature
            e.elev[1] = _adjust_curv(vp.row, vp.col, i, j, e.elev[1],
                                     v_op.do_curv, v_op.ellps_a, v_op.do_refr,
                                     v_op.refr_coef, g_hd.west, g_hd.ew_res,
                                     g_hd.north, g_hd.ns_res, g_hd.proj)

            # write it into the row of data going through the vp
            if i == vp.row:
                data[0][j] = e.elev[1]
                data[1][j] = e.elev[1]
                data[2][j] = e.elev[1]

            # set the vp, and don't insert it into eventlist
            if i == vp.row and j == vp.col:

                # set_viewpoint_elev(vp, e.elev[1] + v_op.obsElev)
                vp.elev = e.elev[1] + v_op.obs_elev

                if v_op.tgt_elev > 0:
                    vp.target_offset = v_op.tgt_elev
                else:
                    vp.target_offset = 0.0

                _set_visibility(visibility_grid, i, j, 180)
                continue

            # if point is outside maxDist, do NOT include it as an event
            if _outside_max_dist(vp.row, vp.col, g_hd.west, g_hd.ew_res,
                                 g_hd.north, g_hd.ns_res, g_hd.proj,
                                 i, j, v_op.max_distance):
                continue

            # if it got here it is not the vp, not NODATA, and
            # within max distance from vp generate its 3 events
            # and insert them

            # get ENTER elevation
            e.type = ENTERING_EVENT
            e.elev[0] = _calc_event_elev(e.type, e.row, e.col, n_rows, n_cols,
                                         vp.row, vp.col, inrast)

            # adjust for curvature
            if v_op.do_curv:
                ay, ax = _calc_event_pos(e.type, e.row, e.col, vp.row, vp.col)
                e.elev[0] = _adjust_curv(vp.row, vp.col, ay, ax, e.elev[0],
                                         v_op.do_curv, v_op.ellps_a,
                                         v_op.do_refr, v_op.refr_coef,
                                         g_hd.west, g_hd.ew_res,
                                         g_hd.north, g_hd.ns_res, g_hd.proj)

            # get EXIT event
            e.type = EXITING_EVENT
            e.elev[2] = _calc_event_elev(e.type, e.row, e.col, n_rows, n_cols,
                                         vp.row, vp.col, inrast)

            # adjust for curvature
            if v_op.do_curv:
                ay, ax = _calc_event_pos(e.type, e.row, e.col, vp.row, vp.col)
                e.elev[2] = _adjust_curv(vp.row, vp.col, ay, ax, e.elev[2],
                                         v_op.do_curv, v_op.ellps_a,
                                         v_op.do_refr, v_op.refr_coef,
                                         g_hd.west, g_hd.ew_res,
                                         g_hd.north, g_hd.ns_res, g_hd.proj)

            # write adjusted elevation into the row of data
            # going through the vp
            if i == vp.row:
                data[0][j] = e.elev[0]
                data[1][j] = e.elev[1]
                data[2][j] = e.elev[2]

            # put event into event list
            e.type = ENTERING_EVENT

            ay, ax = _calc_event_pos(e.type, e.row, e.col, vp.row, vp.col)
            e.ang = _calculate_angle(ax, ay, vp.col, vp.row)
            tmp_event = Event(e.row, e.col, e.elev, e.ang, e.type)
            event_list.append(tmp_event)

            e.type = CENTER_EVENT
            ay, ax = _calc_event_pos(e.type, e.row, e.col, vp.row, vp.col)
            e.ang = _calculate_angle(ax, ay, vp.col, vp.row)
            tmp_event = Event(e.row, e.col, e.elev, e.ang, e.type)
            event_list.append(tmp_event)

            e.type = EXITING_EVENT
            ay, ax = _calc_event_pos(e.type, e.row, e.col, vp.row, vp.col)
            e.ang = _calculate_angle(ax, ay, vp.col, vp.row)
            tmp_event = Event(e.row, e.col, e.elev, e.ang, e.type)
            event_list.append(tmp_event)

    return


@jit(nb.i8(nb.f8, nb.f8), nopython=True)
def _compare(a, b):

    if a < b:
        return -1
    if a > b:
        return 1
    return 0


def _create_tree_node(val, color=RB_RED):
    # Create a TreeNode using given TreeValue

    # every node has null nodes as children initially, create one such object
    # for easy management
    val.max_grad = SMALLEST_GRAD
    ret = TreeNode(tree_val=val, color=color, left=NIL, right=NIL, parent=NIL)
    return ret


def _create_status_struct():
    # Create and initialize the status struct.
    # return a Tree object with a dummy root.

    key = 0
    gradient = [SMALLEST_GRAD, SMALLEST_GRAD, SMALLEST_GRAD]
    ang = [0, 0, 0]
    max_grad = SMALLEST_GRAD
    tv = TreeValue(key=key, gradient=gradient, ang=ang, max_grad=max_grad)

    root = _create_tree_node(val=tv, color=RB_BLACK)

    status_struct = Tree(root=root)

    return status_struct


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

def _viewshed(raster, vp, v_op, g_hd):

    # create the visibility grid of the sizes specified in the header
    visibility_grid = np.empty(shape=(g_hd.rows, g_hd.cols), dtype=np.float64)
    # set everything initially invisible
    visibility_grid.fill(INVISIBLE)

    n_rows, n_cols = raster.shape
    data = np.zeros(shape=(3, n_cols), dtype=np.float64)

    # construct the event list corresponding to the given input file and vp;
    # this creates an array of all the cells on the same row as the vp
    event_list = []

    _init_event_list(event_list=event_list, raster=raster, vp=vp, v_op=v_op,
                     g_hd=g_hd, data=data, visibility_grid=visibility_grid)

    # sort the events radially by ang
    # s = timer()
    event_list.sort(key=lambda x: x.ang, reverse=False)
    # e = timer()
    # print("sort time ", e - s)

    # create the status structure
    status_struct = _create_status_struct()

    # Put cells that are initially on the sweepline into status structure

    for i in range(vp.col + 1, g_hd.cols):
        status_node = StatusNode(row=vp.row, col=i)
        e = Event(row=vp.row, col=i)
        e.elev[0] = data[0][i]
        e.elev[1] = data[1][i]
        e.elev[2] = data[2][i]

        if (not _is_null(data[1][i])) and \
                (not _outside_max_dist(vp.row, vp.col, g_hd.west, g_hd.ew_res,
                                       g_hd.north, g_hd.ns_res, g_hd.proj,
                                       status_node.row, status_node.col,
                                       v_op.max_distance)):
            # calculate Distance to VP and Gradient,
            # store them into status_node
            # need either 3 elevation values or
            # 3 gradients calculated from 3 elevation values
            # need also 3 angs
            e.type = ENTERING_EVENT
            ay, ax = _calc_event_pos(e.type, e.row, e.col, vp.row, vp.col)
            status_node.ang[0] = _calculate_angle(ax, ay, vp.col, vp.row)
            status_node.grad[0] = _calc_event_grad(ay, ax, e.elev[0],
                                                   vp.row, vp.col, vp.elev,
                                                   g_hd.west, g_hd.ew_res,
                                                   g_hd.north, g_hd.ns_res,
                                                   g_hd.proj)

            e.type = CENTER_EVENT
            ay, ax = _calc_event_pos(e.type, e.row, e.col, vp.row, vp.col)
            status_node.ang[1] = _calculate_angle(ax, ay, vp.col, vp.row)
            status_node.dist_to_viewpoint, status_node.grad[1] = \
                _calc_dist_n_grad(status_node.row, status_node.col, e.elev[1],
                                  vp.row, vp.col, vp.elev, g_hd.west,
                                  g_hd.ew_res, g_hd.north, g_hd.ns_res,
                                  g_hd.proj)

            e.type = EXITING_EVENT
            ay, ax = _calc_event_pos(e.type, e.row, e.col, vp.row, vp.col)
            status_node.ang[2] = _calculate_angle(ax, ay, vp.col, vp.row)
            # _calc_event_grad(status_node, 2, ay, ax, e.elev[2], vp, g_hd)
            status_node.grad[2] = _calc_event_grad(ay, ax, e.elev[2],
                                                   vp.row, vp.col, vp.elev,
                                                   g_hd.west, g_hd.ew_res,
                                                   g_hd.north, g_hd.ns_res,
                                                   g_hd.proj)

            assert status_node.ang[1] == 0

            if status_node.ang[0] > status_node.ang[1]:
                status_node.ang[0] -= 2 * PI

            # insert sn into the status structure
            _insert_into_status_struct(status_node, status_struct)

    # sweep the event_list

    # number of visible cells
    nvis = 0
    nevents = len(event_list)

    for i in range(nevents):
        # get out one event at a time and process it according to its type
        e = event_list[i]
        status_node = StatusNode(row=e.row, col=e.col)

        # calculate Distance to VP and Gradient
        status_node.dist_to_viewpoint, status_node.grad[1] = \
            _calc_dist_n_grad(status_node.row, status_node.col,
                              e.elev[1] + vp.target_offset,
                              vp.row, vp.col, vp.elev,
                              g_hd.west, g_hd.ew_res, g_hd.north, g_hd.ns_res,
                              g_hd.proj)

        etype = e.type
        if etype == ENTERING_EVENT:
            # insert node into structure

            #  need either 3 elevation values or
            # 	     * 3 gradients calculated from 3 elevation values */
            # 	    /* need also 3 angs */
            ay, ax = _calc_event_pos(e.type, e.row, e.col, vp.row, vp.col)
            status_node.ang[0] = e.ang
            # _calc_event_grad(status_node, 0, ay, ax, e.elev[0], vp, g_hd)
            status_node.grad[0] = _calc_event_grad(ay, ax, e.elev[0],
                                                   vp.row, vp.col, vp.elev,
                                                   g_hd.west, g_hd.ew_res,
                                                   g_hd.north, g_hd.ns_res,
                                                   g_hd.proj)

            e.type = CENTER_EVENT
            ay, ax = _calc_event_pos(e.type, e.row, e.col, vp.row, vp.col)
            status_node.ang[1] = _calculate_angle(ax, ay, vp.col, vp.row)
            status_node.dist_to_viewpoint, status_node.grad[1] = \
                _calc_dist_n_grad(status_node.row, status_node.col, e.elev[1],
                                  vp.row, vp.col, vp.elev, g_hd.west,
                                  g_hd.ew_res, g_hd.north, g_hd.ns_res,
                                  g_hd.proj)

            e.type = EXITING_EVENT
            ay, ax = _calc_event_pos(e.type, e.row, e.col, vp.row, vp.col)
            status_node.ang[2] = _calculate_angle(ax, ay, vp.col, vp.row)
            # _calc_event_grad(status_node, 2, ay, ax, e.elev[2], vp, g_hd)
            status_node.grad[2] = _calc_event_grad(ay, ax, e.elev[2],
                                                   vp.row, vp.col, vp.elev,
                                                   g_hd.west, g_hd.ew_res,
                                                   g_hd.north, g_hd.ns_res,
                                                   g_hd.proj)

            e.type = ENTERING_EVENT

            if e.ang < PI:
                if status_node.ang[0] > status_node.ang[1]:
                    status_node.ang[0] -= 2 * PI
            else:
                if status_node.ang[0] > status_node.ang[1]:
                    status_node.ang[1] += 2 * PI
                    status_node.ang[2] += 2 * PI

            _insert_into_status_struct(status_node, status_struct)

        elif etype == EXITING_EVENT:
            # delete node out of status structure
            status_struct._delete_from_tree(status_node.dist_to_viewpoint)

        elif etype == CENTER_EVENT:
            # calculate visibility

            # consider current ang and gradient
            max = _max_grad_in_status_struct(status_struct,
                                             status_node.dist_to_viewpoint,
                                             e.ang, status_node.grad[1])

            # the point is visible: store its vertical ang
            if max <= status_node.grad[1]:
                vert_ang = _get_vertical_ang(vp.elev,
                                             status_node.dist_to_viewpoint,
                                             e.elev[1] + vp.target_offset)

                _set_visibility(visibility_grid, status_node.row,
                                status_node.col, vert_ang)

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

    viewpoint = ViewPoint(row=y_view, col=x_view)

    view_options = ViewOptions(obs_elev=observer_elev, tgt_elev=target_elev,
                               max_dist=max_distance,
                               do_curv=do_curve, do_refr=do_refr)

    grid_header = GridHeader(width=width, height=height, x_range=x_range,
                             y_range=y_range, proj=proj)

    viewshed_img = _viewshed(raster.values, viewpoint, view_options,
                             grid_header)

    visibility = xarray.DataArray(viewshed_img,
                                  coords=raster.coords,
                                  attrs=raster.attrs,
                                  dims=raster.dims)
    return visibility
