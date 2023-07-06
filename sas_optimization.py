import os
import sys
import json
    
import shape_decomposition as sd

import cv2
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import pyplot as plt

from ortools.linear_solver import pywraplp
from os.path import join
import os
import numpy as np

from shapely.geometry import Polygon
import matplotlib.patches as pat
from descartes.patch import PolygonPatch

import math
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely import ops
from shapely import affinity

import networkx as nx

def extend_line_segment(segment, magnitude):
    first = np.array(segment[0])
    second = np.array(segment[1])
    
    new_first = first + (first-second) / np.linalg.norm(first-second) * magnitude
    new_second = second + (second-first) / np.linalg.norm(second-first) * magnitude
    return(new_first, new_second)


'''

'''
class Part:
    def __init__(self, polygon):
        self.polygon = polygon

        self.left_child = None #smaller
        self.right_child = None #greater
    
    def is_leave(self):
        if not self.left_child or not self.right_child:
            return True
        else:
            return False
    
    def contains(self, cut):
        return self.polygon.contains(LineString(cut)) or self.polygon.intersects(LineString(cut))
    
    
    def __repr__(self):
        bounding_box = self.polygon.exterior.bounds
        img_w = int(bounding_box[2])
        img_h = int(bounding_box[3])
        main_fig, main_ax = plt.subplots(nrows=1, ncols=1, num='Layout', figsize=(img_w/100, img_h/100))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
        #main_ax.axis(True)
        main_ax.invert_yaxis()
        main_ax.imshow(255 * np.ones((img_h, img_w, 3), np.uint8), origin='lower')
        #exterior = np.array(polygon.exterior.coords, dtype='int32')
        #interior = [np.array(interior.coords, dtype='int32') for interior in list(polygon.interiors)]
        #lyr_cnt = main_ax.add_patch(pat.Polygon(exterior, closed=True, color='black', fill=False, ls='-', lw=1, zorder=1))
        patch1 = PolygonPatch(self.polygon, fc='#009100', alpha=0.5, zorder=2)
        lyr_cnt = main_ax.add_patch(patch1)
        return ""

class Partition:
    def __init__(self, polygon):
        self.root = Part(polygon)
        #self.tolerance = math.pi / 12.0
        self.cut_list = []
        
    def add_cut(self, cut_vector):
        #print("cut added")
        # keep track of all the cuts
        if self.root.contains(cut_vector):
            self.cut_list.append(cut_vector)
        self._add_cut(cut_vector, self.root)
        
            
    def _add_cut(self, cut_vector, cur_node):
        if cur_node.contains(cut_vector):
            if cur_node.left_child and cur_node.right_child:
                if cur_node.left_child.contains(cut_vector):
                    self._add_cut(cut_vector, cur_node.left_child)
                else:
                    self._add_cut(cut_vector, cur_node.right_child)
            else:
                #print(len())
                children = list(ops.split(cur_node.polygon, LineString([Point(p) for p in extend_line_segment(cut_vector, 3)])))
                if len(children) < 2:
                    return
                children.sort(key=lambda x:-x.area)
                cur_node.left_child = Part(children[0])
                cur_node.right_child = Part(children[1])
        else:
            print("Invalid cut")

    def list_leaves(self):
        return self._list_leaves(self.root)
        
    def _list_leaves(self, cur_node):
        if cur_node.is_leave():
            return [cur_node]
        else:
            return self._list_leaves(cur_node.left_child) + self._list_leaves(cur_node.right_child)
    
    '''
    Transform the partition (shapely polygons) into matrix representation (labeled images)
    for evaluation purposes.
    
    matrix_scale is the minimum axis of the matrix
    '''
    def matrix_representation(self, matrix_scale=150):
        partitions = self.list_leaves()
        bounding_box = self.root.polygon.exterior.bounds
        width = int(math.ceil(bounding_box[2])-math.floor(bounding_box[0]))
        height = int(math.ceil(bounding_box[3])-math.floor(bounding_box[1]))
        
        # Rescale the dimension for faster computation
        if matrix_scale < width < height:
            new_width = matrix_scale
            new_height = int(matrix_scale/width*height)
        elif matrix_scale < height < width:
            new_width = int(matrix_scale/height*width)
            new_height = matrix_scale
        else:
            new_width = width
            new_height = height

        initial = np.zeros((new_height, new_width))
        for x in range(new_width):
            for y in range(new_height):
                i = 1
                for partition in partitions:
                    if (partition.polygon.contains(Point(math.floor(bounding_box[0]) + x/new_width*width, math.floor(bounding_box[1]) + y/new_height*height))):
                        initial[y, x] = i
                        break
                    i+=1
        return initial
    
    def render_partition(self):
        partitions = self.list_leaves()
        length = len(partitions)
        bounding_box = self.root.polygon.exterior.bounds
        
        cmap = matplotlib.cm.get_cmap("jet", length)
        cmaplist = [cmap(i) for i in range(cmap.N)]
        img_w = int(bounding_box[2])
        img_h = int(bounding_box[3])
        main_fig, main_ax = plt.subplots(nrows=1, ncols=1, num='Layout', figsize=(img_w/100, img_h/100))
        #main_ax.set_axis_off()
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
        main_ax.axis(False)
        main_ax.invert_yaxis()
        main_ax.imshow(255 * np.ones((img_h, img_w, 3), np.uint8), origin='lower')
        #exterior = np.array(polygon.exterior.coords, dtype='int32')
        #interior = [np.array(interior.coords, dtype='int32') for interior in list(polygon.interiors)]
        #lyr_cnt = main_ax.add_patch(pat.Polygon(exterior, closed=True, color='black', fill=False, ls='-', lw=1, zorder=1))
        for i, partition in enumerate(partitions):
            patch1 = PolygonPatch(partition.polygon, fc=cmaplist[i], alpha=0.5, zorder=2)
            lyr_cnt = main_ax.add_patch(patch1)
    
def build_medial_graph(multilinestring, line_labels, distance, small_branch = 8, connecting_mode=False):
    G = nx.Graph()
    if multilinestring.geom_type == 'MultiLineString':
        for line in multilinestring:
            for i in range(len(line.coords)-1):
                x_1, y_1 = line.coords[i][0], line.coords[i][1]
                row_1, col_1 = sd.xy2rowcol(x_1, y_1, distance.shape[0])
                x_2, y_2 = line.coords[i+1][0], line.coords[i+1][1]
                row_2, col_2 = sd.xy2rowcol(x_2, y_2, distance.shape[0])
                hash_1 = hash((x_1, y_1))
                hash_2 = hash((x_2, y_2))
                G.add_node(hash_1, x=x_1, y=y_1, distance=distance[row_1,col_1])
                G.add_node(hash_2, x=x_2, y=y_2, distance=distance[row_2,col_2])
                G.add_edge(hash_1, hash_2, weight = math.sqrt((x_1-x_2)**2+(y_1-y_2)**2))
        if connecting_mode:
        # connecting graph in the same component
            labels = set(line_labels)
            for label in labels:
                current_lines = [multilinestring[i] for i, line_label in enumerate(line_labels) if line_label == label]

                for i in range(len(current_lines)-1):
                    j_acc = -1
                    d_acc = np.inf
                    for j in range(i+1, len(current_lines)):
                        l1, l2, d = merge(list(current_lines[i].coords), list(current_lines[j].coords), return_distance=True)
                        if d < d_acc:
                            j_acc = j
                    l1, l2 = merge(list(current_lines[i].coords), list(current_lines[j_acc].coords))
                    hash_1 = hash((l1[0], l1[1]))
                    hash_2 = hash((l2[0], l2[1]))
                    G.add_edge(hash_1, hash_2, weight = math.sqrt((l1[0]-l2[0])**2+(l1[1]-l2[1])**2))
                    
    elif multilinestring.geom_type == 'LineString':
        for i in range(len(multilinestring.coords)-1):
                x_1, y_1 = multilinestring.coords[i][0], multilinestring.coords[i][1]
                row_1, col_1 = sd.xy2rowcol(x_1, y_1, distance.shape[0])
                x_2, y_2 = multilinestring.coords[i+1][0], multilinestring.coords[i+1][1]
                row_2, col_2 = sd.xy2rowcol(x_2, y_2, distance.shape[0])
                hash_1 = hash((x_1, y_1))
                hash_2 = hash((x_2, y_2))
                G.add_node(hash_1, x=x_1, y=y_1, distance=distance[row_1,col_1])
                G.add_node(hash_2, x=x_2, y=y_2, distance=distance[row_2,col_2])
                G.add_edge(hash_1, hash_2, weight = math.sqrt((x_1-x_2)**2+(y_1-y_2)**2))
    
    # Remove small branches
    G.remove_edges_from(nx.selfloop_edges(G)) # remove self-loops for possible error
    branchings = [x for x in G.nodes() if G.degree(x)>=3]
    
    for branching in branchings:
        # check if the branching is deleted by previous runs
        if branching in G.nodes():
            cut_vertices = list(G.neighbors(branching))
            for cut_vertex in cut_vertices:
                temp = G.copy()
                temp.remove_node(branching)
                connected = nx.dfs_tree(temp, source=cut_vertex).to_undirected()
                if len(connected.nodes) < small_branch:
                    G.remove_nodes_from(list(connected.nodes))
    
    return G

def chord_residual(vertex_1, vertex_2, boundary_linestring):
    euclidean_distance = math.sqrt((vertex_1[0] - vertex_2[0]) ** 2 + (vertex_1[1] - vertex_2[1]) ** 2)
    min_arc = sd.minimal_arc(vertex_1, vertex_2, boundary_linestring)
    
    return min_arc - euclidean_distance

def find_center(polygon, G, boundary_vertices):
    ''' visualization
    sd.plot_polygon(polygon, width, height)
    if counter % 20 == 0:
            plt.text(medial_vertex_x, medial_vertex_y, "{ps:.1f}".format(ps =cr), fontsize=10)
    '''
    counter = 0
    center = 0
    current_chord_residual = 0
    for v in G.nodes:
        counter += 1
        medial_vertex_x = G.nodes[v]['x']
        medial_vertex_y = G.nodes[v]['y']
        medial_vertex_distance = G.nodes[v]['distance']
        projections = sd.find_projection_pair(boundary_vertices, medial_vertex_x, medial_vertex_y, medial_vertex_distance)
        if len(projections) < 2: # ignore single projection
            continue
        cr = chord_residual((list(boundary_vertices.coords)[projections[0]][0], 
                             list(boundary_vertices.coords)[projections[0]][1]),
                            (list(boundary_vertices.coords)[projections[1]][0], 
                             list(boundary_vertices.coords)[projections[1]][1]),
                            boundary_vertices)
        if cr > current_chord_residual:
            center = v
            current_chord_residual = cr
        
    return center

'''
Find the closest point on the medial axis from (x, y).
'''
def graph_projection(x, y, G):
    current_projection = list(G.nodes)[0]
    current_distance = math.inf
    for v in G.nodes:
        gx = G.nodes[v]['x']
        gy = G.nodes[v]['y']
        distance = math.sqrt((gx-x)**2+(gy-y)**2)
        if distance < current_distance:
            current_distance = distance
            current_projection = v
    return current_projection, current_distance

'''
distance from (x, y) to center
'''
def distance_to_center(x, y, center_id, G):
    closest_medial_axis_point, distance_to_ma = graph_projection(x, y, G)
    try:
        distance_along_medial_axis = nx.dijkstra_path_length(G, center_id, closest_medial_axis_point, weight='weight')
    except: # if not reachable
        distance_along_medial_axis = math.sqrt((G.nodes[center_id]['x']-x)**2+(G.nodes[center_id]['y']-y)**2)
    return distance_along_medial_axis + distance_to_ma

def patch_to_center(patch, center_id, G):
    return distance_to_center(patch.polygon.centroid.x, patch.polygon.centroid.y, center_id, G)

0# ax + by = c
def get_half_plane(point1, point2, interior_point):
    a = (point1[1] - point2[1])
    b = (point2[0] - point1[0])
    c = point2[0] * point1[1] - point1[0] * point2[1]

    return HalfPlane(a, b, c, interior_point)

class HalfPlane:
    def __init__(self, a, b, c, interior_point):
        # ax + by = c
        self.a = a
        self.b = b
        self.c = c
        self.interior_point = interior_point
        
    def constraint_equation(self, x, y):
        if self.a * self.interior_point[0] + self.b * self.interior_point[1] > self.c:
            return self.a * x + self.b * y >= self.c
        else:
            return self.a * x + self.b * y <= self.c
        
    #def evaluate(self, x, y):
    #    return y - self.slope * x
    
    def __repr__(self):
        return '{0:+.02f}'.format(self.a) + "* x + " + '{0:+.02f}'.format(self.b) + " * y = " + '{0:.02f}'.format(self.c)

'''
Find the maximum area axis-aligned fix-aspect-ratio rectangle
aspect_ratio: defined in terms of width / height

return: (rectangle coordinates: (x1, x2, y1, y2), area)
'''
def min_rec(polygon, aspect_ratio, interior_point):
    solver = pywraplp.Solver.CreateSolver('GLOP')

    # four coordinates of the rectangle
    x1 = solver.NumVar(-solver.infinity(), solver.infinity(), 'x1')
    x2 = solver.NumVar(-solver.infinity(), solver.infinity(), 'x2')
    y1 = solver.NumVar(-solver.infinity(), solver.infinity(), 'y1')
    y2 = solver.NumVar(-solver.infinity(), solver.infinity(), 'y2')

    for x in [x1, x2]:
        for y in [y1, y2]:
            for i in range(len(polygon)):
                polygon_constraint1 = get_half_plane(polygon[i], polygon[(i+1)%len(polygon)], interior_point).constraint_equation(x1, y1)
                polygon_constraint2 = get_half_plane(polygon[i], polygon[(i+1)%len(polygon)], interior_point).constraint_equation(x1, y2)
                polygon_constraint3 = get_half_plane(polygon[i], polygon[(i+1)%len(polygon)], interior_point).constraint_equation(x2, y1)
                polygon_constraint4 = get_half_plane(polygon[i], polygon[(i+1)%len(polygon)], interior_point).constraint_equation(x2, y2)
                # inside polygon
                solver.Add(polygon_constraint1)
                solver.Add(polygon_constraint2)
                solver.Add(polygon_constraint3)
                solver.Add(polygon_constraint4)

    # aspect ratio
    solver.Add((x2-x1) == aspect_ratio * (y2 - y1))
    solver.Maximize(x2 - x1)

    status = solver.Solve()
    # [END solve]

    # [START print_solution]
    if status == pywraplp.Solver.OPTIMAL:
        #print('Solution:')
        #print('Objective value =', solver.Objective().Value())
        #print('x1 =', x1.solution_value())
        #print('y1 =', y1.solution_value())
        #print('x2 =', x2.solution_value())
        #print('y2 =', y2.solution_value())
        return [x1.solution_value(),
                x2.solution_value(),
                y1.solution_value(),
                y2.solution_value()], (x2.solution_value() - x1.solution_value()) * (y2.solution_value() - y1.solution_value())
    else:
        print('The problem does not have an optimal solution.')
        return []

    from shapely import affinity
import math
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely import ops
from descartes.patch import PolygonPatch

import numpy as np
import math

'''
Counter clockwise is positive. For this case, vector_2 should be x direction unit vector
'''
def vector_angle(vector_1, vector_2):
    vector_1 = vector_1 / np.linalg.norm(vector_1)
    vector_2 = vector_2 / np.linalg.norm(vector_2)

    ang = np.arccos(np.clip(np.dot(vector_1, vector_2), -1, 1))
    cross_product = np.cross(vector_1, vector_2)
    
    return ang if cross_product >= 0 else -ang

'''
Data structure representing slicing tree
Get the best scores for this TreeNode
tree_node: TreeNode
cuts: a list of two LineStrings correnspond to Axial and Crosswise cuts for this TreeNode (Axial first)
ars: two aspect ratios defined in terms of width / height

return max_value, configuration for this node
'''
class TreeNode:
    '''
    polygon: Polygon representating this node
    type: type of slicing associated with this node, N(unset), A(axial), C(crosswise)
    left_child: TreeNode, recursive data structure 
    right_child: TreeNdoe, recursive data structure
    '''
    def __init__(self):
        self.polygon = None
        self.type = "N"
        self.configuration = -1
        self.left_child = None # Image 0
        self.right_child = None # Image 1
        self.assignment = {"id": -1, "aspect_ratio": -0.1, "coord":[]}
        self.cut = []
        
    def get_axial(self, medial_axis):
        tangent, projection = medial_axis_tangent(medial_axis, Point(self.centroid()[0], self.centroid()[1]))
        return tangent
        
    def get_crosswise(self, medial_axis):
        tangent, projection = medial_axis_tangent(medial_axis, Point(self.centroid()[0], self.centroid()[1]))
        perpendicular1 = np.array([tangent[1], -tangent[0]])
        perpendicular2 = np.array([-tangent[1], tangent[0]])
        if np.cross(tangent, perpendicular1) > 0:
            return perpendicular1
        else:
            return perpendicular2
        
    def score(self):
        if self.is_leave():
            evaluation(this, )
    
    def is_leaf(self):
        if not self.left_child or not self.right_child:
            return True
        else:
            return False
    
    def contains(self, cut):
        return self.polygon.contains(LineString(cut)) or self.polygon.intersects(LineString(cut))
    
    def area(self):
        return self.polygon.area
    
    def centroid(self):
        return self.polygon.centroid.coords[0]
    
    def get_size(self, direction):
        angle = vector_angle((1.0, 0.0), direction)*180/math.pi
        aligned = affinity.rotate(self.polygon, -angle, (0, 0))
        bounding_box = aligned.bounds
        return bounding_box[2] - bounding_box[0], bounding_box[3] - bounding_box[1]
    
    '''
    Get the current height of this node in the tree
    '''
    def get_height(self):
        if self.is_leaf():
            return 0
        else:
            return max(1 + self.left_child.get_height(), 1 + self.right_child.get_height())
        
    def get_num(self):
        if self.is_leaf():
            return 1
        else:
            return self.left_child.get_num() + self.right_child.get_num()
    
    '''
    For visualization in jupyter notebook
    '''
    def __repr__(self):
        if self.polygon:
            bounding_box = self.polygon.exterior.bounds
            img_w = int(bounding_box[2])
            img_h = int(bounding_box[3])
            main_fig, main_ax = plt.subplots(nrows=1, ncols=1, num='Layout', figsize=(img_w/100, img_h/100))
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
            #main_ax.axis(True)
            main_ax.invert_yaxis()
            main_ax.imshow(255 * np.ones((img_h, img_w, 3), np.uint8), origin='lower')
            #exterior = np.array(polygon.exterior.coords, dtype='int32')
            #interior = [np.array(interior.coords, dtype='int32') for interior in list(polygon.interiors)]
            #lyr_cnt = main_ax.add_patch(pat.Polygon(exterior, closed=True, color='black', fill=False, ls='-', lw=1, zorder=1))
            patch1 = PolygonPatch(self.polygon, fc='#009100', alpha=0.5, zorder=2)
            lyr_cnt = main_ax.add_patch(patch1)
            return ""
        else:
            return str(self.get_num())

    
def centroid_cut(centroid, directions, magnitude):
    new_first = np.array(centroid) + np.array(directions) / np.linalg.norm(np.array(directions)) * magnitude
    #print(np.linalg.norm(np.array(directions)) * magnitude)
    new_second = np.array(centroid) - np.array(directions) / np.linalg.norm(np.array(directions)) * magnitude
    return(new_first, new_second)


'''
When finding projection on medial axis, add the extreme point there will be 
error. The nearest point is not actually on the multilinestring
'''
from shapely.ops import nearest_points

def medial_axis_tangent(multilinestring, point):
    p1, p2 = nearest_points(multilinestring, point)
    if multilinestring.contains(p1):
        location = p1
    else:
        # deal with some corner case
        if multilinestring.contains(Point(p1.x+1, p1.y)):
            location =  Point(p1.x+1, p1.y)
        elif multilinestring.contains(Point(p1.x, p1.y+1)):
            location = Point(p1.x, p1.y+1)
        elif multilinestring.contains(Point(p1.x-1, p1.y)):
            location = Point(p1.x-1, p1.y)
        elif multilinestring.contains(Point(p1.x, p1.y-1)):
            location = Point(p1.x, p1.y-1)
        elif multilinestring.contains(Point(p1.x+1, p1.y+1)):
            location = Point(p1.x+1, p1.y+1)
        elif multilinestring.contains(Point(p1.x+1, p1.y-1)):
            location = Point(p1.x+1, p1.y-1)
        elif multilinestring.contains(Point(p1.x-1, p1.y+1)):
            location = Point(p1.x-1, p1.y+1)
        else:
            location = Point(p1.x-1, p1.y-1)
            
    linear_coordinate = multilinestring.project(location)
    
    # projection point
    projection = multilinestring.interpolate(linear_coordinate).coords[0]
    
    # Get reference points for tangent vector calculation
    reference1 = multilinestring.interpolate(linear_coordinate-5).coords[0]
    reference2 = multilinestring.interpolate(linear_coordinate+5).coords[0]

    # if reference points not exist (corner case)
    if Point(reference1).distance(Point(projection)) > 8.0:
        reference1 = projection
    elif Point(reference2).distance(Point(projection)) > 8.0:
        reference2 = projection

    tangent = np.array([reference1[0] - reference2[0], reference1[1] - reference2[1]])
    tangent = tangent / np.linalg.norm(tangent)
    if tangent[0] < 0:
        tangent = tangent * -1.0
    return tangent, projection

'''
root: a TreeNode representing the root of the tree
images: [{"id": 0, "aspect_ratio": 1.1}, ...]
'''
'''
Balance the tree based on height, random if tied
'''
class BalancedStrategy:
    def choose(self, node):
        if node.left_child.get_height() > node.right_child.get_height():
            return "right"
        elif node.left_child.get_height() < node.right_child.get_height():
            return "left"
        else:
            if random.random() > 0.5:
                return "left"
            else:
                return "right"

'''
Balance the tree based on height, random if tied
'''
class UnbalancedStrategy:
    def choose(self, node):
        if node.left_child.get_height() > node.right_child.get_height():
            if random.random() > 0.3:
                return "left"
            else:
                return "right"
        elif node.left_child.get_height() < node.right_child.get_height():
            if random.random() > 0.3:
                return "right"
            else:
                return "left"
        else:
            if random.random() > 0.5:
                return "right"
            else:
                return "left"

import random
class RandomStrategy:
    #def __init__(self):
    def choose(self, node):
        if random.random() > 0.5:
            return "left"
        else:
            return "right"
            
            
'''
number_of_leaf_node: number of leaf node
'''
def tree_initialization(number_of_leaf_node, balanced=True, fix_seed = False):
    if fix_seed:
        random.seed(10)
    def insert(tree_node, strategy):
        if not tree_node.is_leaf():
            if strategy.choose(tree_node) == "left":
                insert(tree_node.left_child, strategy)
            else:
                insert(tree_node.right_child, strategy)
        else:
            tree_node.left_child = TreeNode()
            tree_node.right_child = TreeNode()
        return tree_node
    
    root = TreeNode()
    ba = BalancedStrategy()
    ub = UnbalancedStrategy()
    for i in range(number_of_leaf_node-1):
        if balanced:
            insert(root, ba)
        else:
            insert(root, ub)
    return root



def leaf_elevation_summary(root):
    def _leaf_depth(tree_node, cur_depth, total_height):
        if tree_node.is_leaf():
            return [total_height-cur_depth]
        else:
            return _leaf_depth(tree_node.left_child, cur_depth + 1, total_height) + _leaf_depth(tree_node.right_child, cur_depth+1, total_height)
        
    def summarize(occurance):
        summary = {}
        for d in occurance:
            if d in summary:
                summary[d] += 1
            else:
                summary[d] = 1

        return summary
    
    total_height = root.get_height()
    return  summarize(_leaf_depth(root, 0, total_height))

'''
Sum two dictionary according to their keys
'''
def sum_dict(dict1, dict2):
    return {k: dict1.get(k, 0) + dict2.get(k, 0) for k in set(dict1) | set(dict2)}
        
'''
Assign image sequentially from least depth to largest depth
Return {depth: [image1, image2, ...]}
e.g.
{3: [{'id': 0, 'aspect_ratio': 1.4},
  {'id': 1, 'aspect_ratio': 0.8},
  {'id': 2, 'aspect_ratio': 0.9},
  {'id': 3, 'aspect_ratio': 1.1}],
 4: [{'id': 4, 'aspect_ratio': 1.6},
  {'id': 5, 'aspect_ratio': 1.1},
  {'id': 0, 'aspect_ratio': 1.4},
  {'id': 1, 'aspect_ratio': 0.8},
  {'id': 2, 'aspect_ratio': 0.9},
  {'id': 3, 'aspect_ratio': 1.1},
  {'id': 4, 'aspect_ratio': 1.6},
  {'id': 5, 'aspect_ratio': 1.1}]}
'''
def calculate_image_assignment(images, summary):
    summary_copy = summary.copy()
    assignment = {}
    keys = sorted(summary.keys(), reverse=True)
    i = 0
    for image in images:
        if summary_copy[keys[i]] == 0:
            i = i + 1
            
        summary_copy[keys[i]] = summary_copy[keys[i]] - 1
        if keys[i] in assignment: # check key exists
            assignment[keys[i]].append(image)
        else:
            assignment[keys[i]] = [image]
    return assignment

def tree_assign_image(tree_node, assignment):
    image_assignment = assignment.copy()
    def traverse(tn, depth):
        if tn.is_leaf():
            tn.assignment = image_assignment[depth].pop()
        else:
            traverse(tn.left_child, depth + 1)
            traverse(tn.right_child, depth + 1)
    traverse(tree_node, 0)

def visualize(tree_node):
    if tree_node.is_leaf():
        return str(tree_node.assignment['id'])
    else:
        return str(tree_node.assignment['id']) + "(" + visualize(tree_node.left_child) + ", " + visualize(tree_node.right_child) + ")"

def _list_leaves(cur_node):
    if cur_node.is_leaf():
        return [cur_node]
    else:
        return _list_leaves(cur_node.left_child) + _list_leaves(cur_node.right_child)

def extract_geometry(tree_node):
    def _list_inner(cur_node):
        if cur_node.is_leaf():
            return []
        else:
            return [cur_node] + _list_inner(cur_node.left_child) + _list_inner(cur_node.right_child)
        
    nodes = _list_leaves(tree_node)
    inner = _list_inner(tree_node)
    polygons = [list(node.polygon.exterior.coords) for node in nodes]
    cuts = [node.cut for node in inner]
    boxes = [node.assignment['coord'] for node in nodes]
    assigned_images = [node.assignment['id'] for node in nodes]
    return polygons, boxes, assigned_images, cuts

'''
Heuristically initialize the slicing tree to reduce the search space
'''
def heuristic_initialization(cur_node, medial_axis, depth):
    if depth == 0 or cur_node.is_leaf():
        return
    else:
        center = cur_node.centroid()
        polygon_dimensions = cur_node.get_size(cur_node.get_axial(medial_axis))
        if polygon_dimensions[0] >=  polygon_dimensions[1]:
            cur_node.configuration = 2
            cut_crosswise = LineString([Point(p) for p in centroid_cut(cur_node.centroid(), cur_node.get_crosswise(medial_axis), polygon_dimensions[1])])
            a = list(ops.split(cur_node.polygon, cut_crosswise))
            a.sort(key=lambda x:-x.area)
            cur_node.left_child.polygon = a[0]
            heuristic_initialization(cur_node.left_child, medial_axis, depth -1)
            cur_node.right_child.polygon = a[1]
            heuristic_initialization(cur_node.right_child, medial_axis, depth -1)
        else:
            cur_node.configuration = 0
            cut_axial = LineString([Point(p) for p in centroid_cut(cur_node.centroid(), cur_node.get_axial(medial_axis), polygon_dimensions[0])])
            a = list(ops.split(cur_node.polygon, cut_axial))
            a.sort(key=lambda x:-x.area)
            cur_node.left_child.polygon = a[0]
            heuristic_initialization(cur_node.left_child, medial_axis, depth -1)
            cur_node.right_child.polygon = a[1]
            heuristic_initialization(cur_node.right_child, medial_axis, depth -1)
import random
def random_initialization(cur_node, medial_axis, depth):
    if depth == 0 or cur_node.is_leaf():
        return
    else:
        cur_node.configuration = random.randint(0, 3)
        random_initialization(cur_node.left_child, medial_axis, depth -1)
        random_initialization(cur_node.right_child, medial_axis, depth -1)

import math

def interior_angle(vector_1, vector_2):
    vector_1 = vector_1 / np.linalg.norm(vector_1)
    vector_2 = vector_2 / np.linalg.norm(vector_2)

    ang = np.arccos(np.clip(np.dot(vector_1, vector_2), -1, 1))
    ang = np.abs(ang) if np.cross(vector_1, vector_2) > 0 else 2*np.pi - np.abs(ang)
    return ang * 180 / math.pi

'''
Vertices given in clockwise order. First vertex not repeated
'''
def interior_angles(vertices):
    angles = []
    for i in range(len(vertices)):
        p1 = np.array(vertices[i])
        ref = np.array(vertices[i-1])
        p2 = np.array(vertices[(i+1)%len(vertices)])
        v1 = ref - p1
        v2 = p2 - p1
        angles.append(interior_angle(v1, v2))
    return angles

'''
True: good cell, False: bad cell
'''
def cell_quality(polygon, simplication_threshold=10):
    vertices = list(polygon.simplify(simplication_threshold).exterior.coords)[0:-1]
    side = len(vertices)
    angles = interior_angles(vertices)
    sharp_angle = any(angle < 35 for angle in angles)
    if side <= 3:# or sharp_angle:
        return False
    else:
        return True

'''
Recursively solve for the optimal slicing tree
'''
def get_optimal(tree_node, medial_axis):
    if tree_node.is_leaf():
        # Handling Axial case
        convex = tree_node.polygon.convex_hull.simplify(10) # Simplify the geometry to speed up
        quality_cell = cell_quality(tree_node.polygon)
        optimal = min_rec(convex.exterior.coords, tree_node.assignment['aspect_ratio'], list(convex.representative_point().coords)[0])
        decision = TreeNode()
        decision.configuration = -1
        decision.polygon = tree_node.polygon
        decision.assignment = tree_node.assignment.copy()
        decision.assignment["coord"] = optimal[0]
        if quality_cell:
            return optimal[1], decision
        else:
            return optimal[1]*-0.8, decision # penalty
    else:
        if tree_node.configuration == -1: # If the configuration is not determined yet
            acc = []
            cuts = []
            polygon_dimensions = tree_node.get_size(tree_node.get_axial(medial_axis))
            cut_axial = LineString([Point(p) for p in centroid_cut(tree_node.centroid(), tree_node.get_axial(medial_axis), polygon_dimensions[0])])
            cut_crosswise = LineString([Point(p) for p in centroid_cut(tree_node.centroid(), tree_node.get_crosswise(medial_axis), polygon_dimensions[1])])
            # Handling Axial case
            a = list(ops.split(tree_node.polygon, cut_axial))
            a.sort(key=lambda x:-x.area)
            cut = tree_node.polygon.intersection(cut_axial)
            # Deal with small interesecting line segment
            if cut.geom_type == 'MultiLineString':
                lines = [line for line in cut]
                lines.sort(key=lambda x:-x.length)
                cut = lines[0]
            cuts.append(cut)
            
            # Axial case 1
            tree_node.right_child.polygon = a[0]
            tree_node.left_child.polygon = a[1]
            acc.append((get_optimal(tree_node.left_child, medial_axis), get_optimal(tree_node.right_child, medial_axis)))


            # Axial case 2
            tree_node.right_child.polygon = a[1]
            tree_node.left_child.polygon = a[0]
            acc.append((get_optimal(tree_node.left_child, medial_axis), get_optimal(tree_node.right_child, medial_axis)))

            # Handling Crosswise case
            a = list(ops.split(tree_node.polygon, cut_crosswise))
            a.sort(key=lambda x:-x.area)
            cut = tree_node.polygon.intersection(cut_crosswise)
            # Deal with small interesecting line segment
            if cut.geom_type == 'MultiLineString':
                lines = [line for line in cut]
                lines.sort(key=lambda x:-x.length)
                cut = lines[0]
            cuts.append(cut)
            # Crosswise case 1
            tree_node.right_child.polygon = a[0]
            tree_node.left_child.polygon = a[1]
            acc.append((get_optimal(tree_node.left_child, medial_axis), get_optimal(tree_node.right_child, medial_axis)))


            # Crosswise case 2
            tree_node.right_child.polygon = a[1]
            tree_node.left_child.polygon = a[0]
            acc.append((get_optimal(tree_node.left_child, medial_axis), get_optimal(tree_node.right_child, medial_axis)))

            utility = [pair[0][0] + pair[1][0] for pair in acc]

            decision = TreeNode()
            decision.configuration = np.argmax(utility)
            decision.polygon = tree_node.polygon
            decision.cut = list(cuts[math.floor(np.argmax(utility)/2)].coords)
            decision.left_child = acc[np.argmax(utility)][0][1]
            decision.right_child = acc[np.argmax(utility)][1][1]

            return max(utility), decision
        else: # If the configuration is predifined (to reduce calculation)
            acc = []
            polygon_dimensions = tree_node.get_size(tree_node.get_axial(medial_axis))
            cut_axial = LineString([Point(p) for p in centroid_cut(tree_node.centroid(), tree_node.get_axial(medial_axis), polygon_dimensions[0])])
            cut_crosswise = LineString([Point(p) for p in centroid_cut(tree_node.centroid(), tree_node.get_crosswise(medial_axis), polygon_dimensions[1])])
            # Handling Axial case
            a = list(ops.split(tree_node.polygon, cut_axial))
            a.sort(key=lambda x:-x.area)
            cut_a = tree_node.polygon.intersection(cut_axial)
            # Deal with small interesecting line segment
            if cut_a.geom_type == 'MultiLineString':
                lines = [line for line in cut_a]
                lines.sort(key=lambda x:-x.length)
                cut_a = lines[0]
            # Axial case 1
            if tree_node.configuration == 0:
                tree_node.right_child.polygon = a[0]
                tree_node.left_child.polygon = a[1]
                optimum = (get_optimal(tree_node.left_child, medial_axis), get_optimal(tree_node.right_child, medial_axis))
                cut = cut_a

            # Axial case 2
            if tree_node.configuration == 1:
                tree_node.right_child.polygon = a[1]
                tree_node.left_child.polygon = a[0]
                optimum = (get_optimal(tree_node.left_child, medial_axis), get_optimal(tree_node.right_child, medial_axis))
                cut = cut_a
            # Handling Crosswise case
            a = list(ops.split(tree_node.polygon, cut_crosswise))
            a.sort(key=lambda x:-x.area)
            cut_c = tree_node.polygon.intersection(cut_crosswise)
            # Deal with small interesecting line segment
            if cut_c.geom_type == 'MultiLineString':
                lines = [line for line in cut_c]
                lines.sort(key=lambda x:-x.length)
                cut_c = lines[0]
            # Crosswise case 1
            if tree_node.configuration == 2:
                tree_node.right_child.polygon = a[0]
                tree_node.left_child.polygon = a[1]
                optimum = (get_optimal(tree_node.left_child, medial_axis), get_optimal(tree_node.right_child, medial_axis))
                cut = cut_c

            # Crosswise case 2
            if tree_node.configuration == 3:
                tree_node.right_child.polygon = a[1]
                tree_node.left_child.polygon = a[0]
                optimum = (get_optimal(tree_node.left_child, medial_axis), get_optimal(tree_node.right_child, medial_axis))
                cut = cut_c

            utility = optimum[0][0] + optimum[1][0]

            decision = TreeNode()
            decision.configuration = tree_node.configuration
            decision.polygon = tree_node.polygon
            decision.cut = list(cut.coords)
            decision.left_child = optimum[0][1]
            decision.right_child = optimum[1][1]

            return utility, decision

import cv2
def load_mask(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return image
'''
Extract foreground pixels' bounding box.
If no foreground pixel or too small, return the whole image as bounding box
'''
def extract_foreground(label):
    total_area = label.shape[0]*label.shape[1]
    foreground = (label==255).astype(int)
    foreground_area = np.sum(foreground)
    foreground_exist = True
    
    if foreground_area > total_area / 200:
        x1 = int(np.min(np.where(foreground)[1]))
        x2 = int(np.max(np.where(foreground)[1]))
        y1 = int(label.shape[0] - np.max(np.where(foreground)[0]))
        y2 = int(label.shape[0] - np.min(np.where(foreground)[0]))
    else:
        foreground_exist = False
        x1 = int(label.shape[1]/10)
        x2 = int(label.shape[1]*9/10)
        y1 = int(label.shape[0]/10)
        y2 = int(label.shape[0]*9/10)
    return x1, x2, y1, y2, foreground_exist

def process_image_for_optimization(image_dict):
    index = 0
    images = []
    for idct in image_dict:
        aspect_ratio = (idct['foreground'][1]-idct['foreground'][0])/(idct['foreground'][3]-idct['foreground'][2])
        images.append({'id': index, 'aspect_ratio': aspect_ratio})
        index = index + 1
    return images

'''
Assign images to tree leaves
'''
def assign_image(tree_node, assignment):
    image_assignment = assignment.copy()
    def traverse(tn, depth):
        if tn.is_leaf():
            tn.assignment = image_assignment[depth].pop()
        else:
            traverse(tn.left_child, depth + 1)
            traverse(tn.right_child, depth + 1)
    traverse(tree_node, 0)

'''
Calculate image per part adjust for small part (zero image patch)

Making sure the sum of count is correct
'''
def adjust_image_per_part(raw_image_per_part, target):
    copy = raw_image_per_part.copy()
    zero_sum = len([count for count in raw_image_per_part if count == 0])
    
    current = sum(copy)
    delta = (current + zero_sum) - target
    # change zero image per part to one (for small patches)
    for i in range(len(copy)):
        if copy[i] == 0:
            copy[i] = 1
    if delta > 0:
        # adjust for the addition of image
        for _ in range(delta):
            max_count = 0
            max_idx = -1
            for i in range(len(copy)):
                if copy[i] > max_count:
                    max_count = copy[i]
                    max_idx = i
            copy[max_idx] -= 1
    elif delta < 0: # if current count is smaller than target (because the rounding precision)        
        for _ in range(-delta):
            max_count = 0
            max_idx = -1
            for i in range(len(copy)):
                if copy[i] > max_count:
                    max_count = copy[i]
                    max_idx = i
            copy[max_idx] += 1
    return copy

'''
Initialize a slicing tree for every part, hence a forest.
'''
def forest_initialization(convex_parts, n, total_area, balanced, multilinestring_int):
    image_per_part = [round(n*convex_part.polygon.area/total_area) for convex_part in convex_parts]
    image_per_part = adjust_image_per_part(image_per_part, n)
    
    forest = []
    forest_summary = {}
    for i, convex_part in enumerate(convex_parts):
        root = tree_initialization(image_per_part[i], balanced=balanced)
        root.polygon = convex_part.polygon
        heuristic_level = root.get_height() - 3
        heuristic_initialization(root, multilinestring_int[0], heuristic_level) # Heuristically determine node configuration
        #random_initialization(root, multilinestring_int[0], 1000) # random initialization
        forest_summary = sum_dict(forest_summary, leaf_elevation_summary(root))
        forest.append(root)

    return forest, forest_summary

'''
Assign image to leaves of all trees based on depth and tree priority
'''

def assign_image(forest, assignment):
    image_assignment = assignment.copy()
    def traverse(tn, depth, tree_height):
        if tn.is_leaf():
            
            tn.assignment = image_assignment[tree_height-depth].pop(0)
        else:
            traverse(tn.left_child, depth + 1, tree_height)
            traverse(tn.right_child, depth + 1, tree_height)
    for tree in forest:
        tree_height = tree.get_height()
        traverse(tree, 0, tree_height)

def forest_optimization(forest, multilinestring_int):
    solution = []
    for tree in forest:
        optima = get_optimal(tree, multilinestring_int[0])
        solution.append(optima)
    return solution


def extract_forest_geometry(forest):
    parts_dict = []

    part_format = {
        "index": 0,
        "coords": [],
        "foreground": []
    }
    
    idx = 0
    mapping = {} # map from parts to image
    cuts = []
    for tree in forest:
        geometry = extract_geometry(tree)
        cuts = cuts + geometry[3]
        for i in range(len(geometry[0])):
            pf = part_format.copy()
            pf["index"] = idx
            pf["coords"] = geometry[0][i]
            pf["foreground"] = geometry[1][i]
            parts_dict.append(pf)
            mapping[idx] = geometry[2][i]
            idx += 1
    return parts_dict, mapping, cuts

def render_matching_result(final_layout, rectangles, assigned_images, width, height, label=False):
    #partitions = self.list_leaves()
    length = len(final_layout)
    #bounding_box = self.root.polygon.exterior.bounds

    cmap = matplotlib.cm.get_cmap("jet", length)
    cmaplist = [cmap(i) for i in range(cmap.N)]
    #img_w = int(bounding_box[2])
    #img_h = int(bounding_box[3])
    main_fig, main_ax = plt.subplots(nrows=1, ncols=1, num='Layout', figsize=(width/50, height/50))
    #main_ax.set_axis_off()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    main_ax.axis(False)
    main_ax.invert_yaxis()
    main_ax.imshow(255 * np.ones((height, width, 3), np.uint8), origin='lower')
    #exterior = np.array(polygon.exterior.coords, dtype='int32')
    #interior = [np.array(interior.coords, dtype='int32') for interior in list(polygon.interiors)]
    #lyr_cnt = main_ax.add_patch(pat.Polygon(exterior, closed=True, color='black', fill=False, ls='-', lw=1, zorder=1))
    for i, partition in enumerate(final_layout):
        patch1 = PolygonPatch(Polygon(partition), fc=cmaplist[i], alpha=0.5, zorder=2)
        lyr_cnt = main_ax.add_patch(patch1)
        
        rec_coords = [(rectangles[i][0], rectangles[i][2]), 
              (rectangles[i][1], rectangles[i][2]),
              (rectangles[i][1], rectangles[i][3]),
              (rectangles[i][0], rectangles[i][3])]
        patch2 = PolygonPatch(Polygon(rec_coords), fc=cmaplist[i], alpha=0.5, zorder=3)
        lyr_cnt = main_ax.add_patch(patch2)
        if label:
            main_ax.text(Polygon(partition).centroid.coords[0][0], Polygon(partition).centroid.coords[0][1], "part "+str(i))
            main_ax.text(Polygon(rec_coords).centroid.coords[0][0], Polygon(rec_coords).centroid.coords[0][1]+20, "image "+str(assigned_images[i]), color="red")

def optimization(input_shape, input_mask_folder, output_dir):
    image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    polygon = sd.generate_canvas_polygon(image)[0]
    with open(join(output_dir, 'final_cut.json')) as f:
        prediction = json.load(f)

    prediction_partition = Partition(polygon)
    for cut in prediction:
        prediction_partition.add_cut(cut)
    prediction_partition.render_partition()
    plt.savefig(join(output_dir, 'raw_parts.png'), bbox_inches='tight')
    medial_interior_input = sd.prepare_for_medial_axis(image, complement=False)
    ma_int = sd.ridge_medial_axis(medial_interior_input, ridge_threshold = 0.39, small_threshold=5)
    multilinestring_int = sd.build_medial_multilinestring(ma_int[0])
    final_medial_vertices_int = sd.redistribute_vertices(multilinestring_int[0], 5)
    convex_parts = prediction_partition.list_leaves()
    G = build_medial_graph(final_medial_vertices_int, multilinestring_int[1], ma_int[1])
    boundary_vertices = sd.redistribute_vertices(LineString(polygon.exterior.coords), 5)
    center_id = find_center(polygon, G, boundary_vertices)
    convex_parts.sort(key=lambda x: -1*patch_to_center(x, center_id, G)) # sort convex patches by its distance to center

    # Read input image collection
    image_ids = [f.split(".")[0] for f in os.listdir(input_mask_folder) if f.endswith('.png')]
    image_dict = []


    image_template = {
        "filename": "",
        "foreground_exists": True,
        "foreground": [],
        "assigned_part": 0
    }

    for image_id in image_ids:
        label = load_mask(join(input_mask_folder, image_id + ".png"))
        x1, x2, y1, y2, foreground_exist = extract_foreground(label)
        it = image_template.copy()
        it["filename"] = image_id + ".jpg"
        it["foreground"] = [x1, x2, y1, y2]
        if not foreground_exist:
            it['foreground_exists'] = False
        image_dict.append(it)


    images = process_image_for_optimization(image_dict)
    print("Total", len(images), "images.")
    ss = forest_initialization(convex_parts, len(images), prediction_partition.root.polygon.area, True, multilinestring_int)
    kk = calculate_image_assignment(images, ss[1])
    assign_image(ss[0], kk)
    result = forest_optimization(ss[0], multilinestring_int)
    forest = [r[1] for r in result]
    geometry = extract_forest_geometry(forest)

    plt.clf()
    render_matching_result([g['coords'] for g in geometry[0]], [g['foreground'] for g in geometry[0]], geometry[1], image.shape[1], image.shape[0], label=True)
    plt.savefig(join(output_dir, 'optimal_layout.png'), bbox_inches='tight')
    # Calculate inverse mapping
    inv_map = {v: k for k, v in geometry[1].items()}
    for k in inv_map:
        image_dict[k]['assigned_part'] = inv_map[k]
        
    cuts = geometry[2] + prediction

    optimization_output = {
        "images": image_dict,
        "parts": geometry[0],
        "width": image.shape[1],
        "height": image.shape[0],
        "cuts": cuts
    }

    with open(join(output_dir, 'slicing_result.json'), 'w') as f:
        json.dump(optimization_output, f)

if __name__ == '__main__':
    input_shape = sys.argv[1]
    input_mask_folder = sys.argv[2]
    output_dir = sys.argv[3]
    optimization(input_shape, input_mask_folder, output_dir)

    