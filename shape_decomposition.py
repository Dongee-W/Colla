import os
import json
from os import path
import cv2
import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from descartes.patch import PolygonPatch
from os import listdir
from os.path import isfile, join
import sys

'''
Convert Input Image in to Contour Polygon. Support polygons with holes and multiple polygons.

Black in assume to be the region of the polygon.
'''
def generate_canvas_polygon(img, complement=False):
    threshold = 127
    area_filter = 0.01
    
    # Convert to gray scale
    if len(img.shape) >= 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = img.shape
    total_area = img_h * img_w
    img = cv2.flip(img, 0) # reverse y direction
    #img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, (255, 255, 255))
    #img = cv2.resize(img, (int(img_w), int(img_h)), cv2.INTER_LINEAR)
    _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    
    if not complement:
        img = cv2.bitwise_not(img) # Invert white and black pixels
    
    # Remove noise
    img = cv2.GaussianBlur(img, (3,3), sigmaX=0.2, borderType=cv2.BORDER_DEFAULT)
    
    # contour format: (x, y) x in the width dimension, y in the height direction with top being zero 
    contour, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    hierarchy = np.reshape(hierarchy, (-1, 4))
    for c in enumerate(hierarchy):
        holes = []
        # parse hiearchy
        if(c[1][2] >= 0 or (c[1][2] == -1 and c[1][3] == -1)):
            next_child = c[1][2] 
            # Extract holes
            while next_child >= 0:
                holes.append(np.squeeze(contour[next_child]))
                next_child = hierarchy[next_child][0]
            # check if contour has at least three points
            #if np.squeeze(contour[c[0]])
            if np.squeeze(contour[c[0]]).shape[0] > 3:
                polygon = Polygon(np.squeeze(contour[c[0]]), holes)
                if polygon.area >= total_area * area_filter:
                    polygons.append(polygon)
    return polygons

def plot_polygon(polygon, img_w, img_h):
    main_fig, main_ax = plt.subplots(nrows=1, ncols=1, num='Layout', figsize=(img_w/100, img_h/100))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    #main_ax.axis(True)
    main_ax.invert_yaxis()
    main_ax.imshow(255 * np.ones((img_h, img_w, 3), np.uint8), origin='lower')
    #exterior = np.array(polygon.exterior.coords, dtype='int32')
    #interior = [np.array(interior.coords, dtype='int32') for interior in list(polygon.interiors)]
    #lyr_cnt = main_ax.add_patch(pat.Polygon(exterior, closed=True, color='black', fill=False, ls='-', lw=1, zorder=1))
    patch1 = PolygonPatch(polygon, fc='#009100', alpha=0.5, zorder=2)
    lyr_cnt = main_ax.add_patch(patch1)
    return

'''
Preprocessing the input image for medial axis extraction
1. Bluring to avoid uncessary noise
2. Thresholding to convert the input into binary images
'''
def prepare_for_medial_axis(img, complement=False):
    threshold = 127
    
    if len(img.shape) >= 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = img.shape
    
    # Scale down if the image is large. Mainly to avoid unnecessary noise
    #if img_h <= img_w:
    #    img = cv2.resize(img, (int(400*img_w/img_h), int(400)), cv2.INTER_CUBIC)
    #else:
    #    img = cv2.resize(img, (int(400), int(400*img_h/img_w)), cv2.INTER_CUBIC)
        
    # Add smoothness to the geometry
    img = cv2.GaussianBlur(img, (5,5), cv2.BORDER_DEFAULT)
    
    _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    if not complement:
        img = cv2.bitwise_not(img) # Invert white and black pixels
    
    return img > 127

from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage import morphology
import scipy
import matplotlib

import math
class MedialAxisGroup:
    def __init__(self, old_index, new_index, vertices):
        self.old_index = old_index
        self.new_index = new_index
        self.vertices = vertices
        
    def __repr__(self):
        return 'MAG('+ str(self.old_index) + "->" + str(self.new_index) + ')'

'''
Utility function for merging two component with closest points
'''
def merge(set1, set2, return_distance=False):
    kdTree = scipy.spatial.KDTree(set1)
    d_acc = np.inf
    nearest_acc = None
    point_acc = None
    for point in set2:
        d, nearest = kdTree.query([point[0], point[1]], k=1)
        if d < d_acc:
            d_acc = d
            nearest_acc = set1[nearest]
            point_acc = point
    if return_distance:
        return nearest_acc, point_acc, d_acc
    else:
        return nearest_acc, point_acc

def detect_ridges(gray, sigma=3.0):
    H_elems = hessian_matrix(gray, sigma=0.1)
    i1, i2 = hessian_matrix_eigvals(H_elems)
    return i1, i2

def ridge_medial_axis(image, ridge_threshold = 0.3, small_threshold=5, component_max_distance=15, background_size=40000):
    distance_map = scipy.ndimage.distance_transform_edt(image, return_indices=True)
    detected_ridges = detect_ridges(distance_map[0])[1]<-ridge_threshold
    
    # Cleanup ridges map by filtering out small objects
    cleanup = np.uint8(morphology.remove_small_objects(detected_ridges, small_threshold))
    
    # Connect broken lines
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    fix_broken = cv2.dilate(cleanup, kernel, iterations = 1)

    # Find 1-pixel wide skeleton after dilation
    skeleton = np.uint8(morphology.skeletonize(fix_broken))
    
    # Remove remaining isolated patches
    second_filtered = np.uint8(morphology.remove_small_objects(skeleton>0, 10, 8))

    # Identify connected component
    cc = cv2.connectedComponentsWithStats(second_filtered, 8, cv2.CV_32S)
    
    medial_axis_group = cc[1].copy()
    components = []
    for i, component in enumerate(cc[2]):
        if component[4]< background_size and component[4]>10:
            components.append(MedialAxisGroup(i, i, np.argwhere(medial_axis_group == i)))
        elif component[4]< background_size:
            medial_axis_group[medial_axis_group == i] = 0
    # Connect nearby components
    for i in range(len(components)):
        for j in range(i+1, len(components)):
            g1, g2, d = merge(components[i].vertices, components[j].vertices, return_distance=True)
            if d<component_max_distance:
                components[j].new_index = components[i].new_index
    
    for component in components:
        if component.old_index != component.new_index:
            group1 = np.argwhere(medial_axis_group == component.old_index)
            group2 = np.argwhere(medial_axis_group == component.new_index)
            g1, g2 = merge(group1, group2)
            connecting = medial_axis_group.copy()
            # connect the closest points from each group
            cv2.line(connecting, (g2[1], g2[0]), (g1[1],g1[0]), color=component.new_index, thickness=1)
            medial_axis_group = connecting
            medial_axis_group[medial_axis_group == component.old_index] = component.new_index
            
    
    return medial_axis_group, distance_map[0]

def plot_medial_axis(contour, medial_axis, distance):
    # Distance to the background for pixels of the skeleton
    #dist_on_skel = distance * skel

    main_fig, main_ax = plt.subplots(nrows=1, ncols=1, num='Layout', figsize=(medial_axis.shape[1]/100, medial_axis.shape[0]/100))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)

    main_ax.matshow(medial_axis)
    main_ax.contour(contour, [0.5], colors='w')

def rowcol2xy(row, col, ymax):
    return int(col), int(ymax - row)

def xy2rowcol(x, y , ymax):
    return int(round(ymax - y, 0)), int(round(x, 0))

import networkx as nx
from pyvis.network import Network

from shapely.geometry import MultiLineString
from shapely import geometry, ops

from shapely.geometry import MultiLineString
from shapely import geometry, ops

# Mixed connectivity: Remove 8-connected neighbors when 4-connected neighbors are present
def mixed_connectivity(neighbors):
    if (0,1) in neighbors:
        if (0,0) in neighbors:
            neighbors.remove((0,0))
        if (0,2) in neighbors:
            neighbors.remove((0,2))
    if (1,0) in neighbors:
        if (0,0) in neighbors:
            neighbors.remove((0,0))
        if (2,0) in neighbors:
            neighbors.remove((2,0))
    if (2,1) in neighbors:
        if (2,0) in neighbors:
            neighbors.remove((2,0))
        if (2,2) in neighbors:
            neighbors.remove((2,2))
    if (1,2) in neighbors:
        if (0,2) in neighbors:
            neighbors.remove((0,2))
        if (2,2) in neighbors:
            neighbors.remove((2,2))
    return neighbors

def build_medial_multilinestring(medial_axis, min_line_length=20):
    medial_points = np.argwhere(medial_axis>0)
    #print(medial_points)
    coords = []
    for medial_point in medial_points:
        r = medial_point[0]
        c = medial_point[1]
        if r>0 and r<medial_axis.shape[0] and c>0 and c<medial_axis.shape[1]:
            copied = np.copy(medial_axis[r-1:(r+1)+1, c-1:(c+1)+1])
            copied[1,1] = False
            neighbor_points = np.argwhere(copied)
            acc = []
            for row, col in neighbor_points:
                acc.append((row, col))
            m_connected = mixed_connectivity(acc)    
            
            neighbors = medial_point + np.array(m_connected) - 1
            x, y = rowcol2xy(r, c, medial_axis.shape[0])

            for neighbor in neighbors:
                neighbor_x, neighbor_y = rowcol2xy(neighbor[0], neighbor[1], medial_axis.shape[0])
                line = ((x, y), (neighbor_x, neighbor_y))
                reverse = ((neighbor_x, neighbor_y), (x, y))
                if line not in coords and reverse not in coords:
                    coords.append(line)
    merged_line = ops.linemerge(MultiLineString(coords))
            
    # output connected component lable for graph generation
    line_label = []
    if merged_line.geom_type == 'MultiLineString':
        for line in merged_line:
            #print(list(line.coords)[0][0]-medial_axis.shape[0] )
            row, col = xy2rowcol(list(line.coords)[0][0], list(line.coords)[0][1] , medial_axis.shape[0])
            line_label.append(medial_axis[row, col])
    elif merged_line.geom_type == 'LineString':
        row, col = xy2rowcol(list(merged_line.coords)[0][0], list(merged_line.coords)[0][1] , medial_axis.shape[0])
        line_label.append(medial_axis[row, col])
    return merged_line, line_label

def plot_multilinestring(lines, line_label, polygon, width, height):
    plot_polygon(polygon, width, height)
    cmap = matplotlib.cm.get_cmap("jet", max(line_label)+1)
    cmaplist = [cmap(i) for i in range(cmap.N)]
    if lines.geom_type == 'MultiLineString':
        for i, line in enumerate(lines):
            x = [point[0] for point in line.coords]
            y = [point[1] for point in line.coords]
            plt.plot(x, y, marker='o', markersize=1, linewidth=.1, c=cmaplist[line_label[i]])
    elif lines.geom_type == 'LineString':
        x = [point[0] for point in lines.coords]
        y = [point[1] for point in lines.coords]
        plt.plot(x, y, marker='o', markersize=1, linewidth=.1, c=cmaplist[line_label[0]])
        
from shapely.geometry import LineString
from shapely.geometry.polygon import LinearRing

'''
Resample the vertices at fixed gap for more uniform representations
'''
def redistribute_vertices(geom, gap):
    if geom.geom_type == 'LineString':
        num_vert = int(round(geom.length / gap))
        if num_vert == 0:
            num_vert = 1
        return LineString(
            [geom.interpolate(float(n) / num_vert, normalized=True)
             for n in range(num_vert+1)])
    elif geom.geom_type == 'LinearRing':
        num_vert = int(round(geom.length / gap))
        if num_vert == 0:
            num_vert = 1
        return LinearRing(
            [geom.interpolate(float(n) / num_vert, normalized=True)
             for n in range(num_vert+1)])
    elif geom.geom_type == 'MultiLineString':
        parts = [redistribute_vertices(part, gap)
                 for part in geom]
        return type(geom)([p for p in parts if not p.is_empty])
    else:
        raise ValueError('unhandled geometry %s', (geom.geom_type,))

def build_medial_graph(multilinestring, line_labels, distance, small_branch = 8, connecting_mode=False):
    G = nx.Graph()
    if multilinestring.geom_type == 'MultiLineString':
        for line in multilinestring:
            for i in range(len(line.coords)-1):
                x_1, y_1 = line.coords[i][0], line.coords[i][1]
                row_1, col_1 = xy2rowcol(x_1, y_1, distance.shape[0])
                x_2, y_2 = line.coords[i+1][0], line.coords[i+1][1]
                row_2, col_2 = xy2rowcol(x_2, y_2, distance.shape[0])
                hash_1 = hash((x_1, y_1))
                hash_2 = hash((x_2, y_2))
                G.add_node(hash_1, x=x_1, y=y_1, distance=distance[row_1,col_1])
                G.add_node(hash_2, x=x_2, y=y_2, distance=distance[row_2,col_2])
                G.add_edge(hash_1, hash_2)
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
                    G.add_edge(hash_1, hash_2)
                    
    elif multilinestring.geom_type == 'LineString':
        for i in range(len(multilinestring.coords)-1):
                x_1, y_1 = multilinestring.coords[i][0], multilinestring.coords[i][1]
                row_1, col_1 = xy2rowcol(x_1, y_1, distance.shape[0])
                x_2, y_2 = multilinestring.coords[i+1][0], multilinestring.coords[i+1][1]
                row_2, col_2 = xy2rowcol(x_2, y_2, distance.shape[0])
                hash_1 = hash((x_1, y_1))
                hash_2 = hash((x_2, y_2))
                G.add_node(hash_1, x=x_1, y=y_1, distance=distance[row_1,col_1])
                G.add_node(hash_2, x=x_2, y=y_2, distance=distance[row_2,col_2])
                G.add_edge(hash_1, hash_2)
    
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
                if cut_vertex in temp.nodes():
                    connected = nx.dfs_tree(temp, source=cut_vertex).to_undirected()
                    if len(connected.nodes) < small_branch:
                        G.remove_nodes_from(list(connected.nodes))
    
    return G


def interactive_graph(G):
    nt = Network('800px', '800px')
    nt.from_nx(G)
    nt.show('nx.html')
    
def find_end_vertices(G, exterior=True):
    acc = []
    for cc in nx.connected_components(G):
        subgraph = G.subgraph(cc).copy()
        end_vertices_candidates = [x for x in subgraph.nodes() if subgraph.degree(x)==1]
        for candidate in end_vertices_candidates:
            distance = subgraph.nodes[candidate]['distance']

            neighbor_distance = subgraph.nodes[list(subgraph.neighbors(candidate))[0]]['distance']
            
            if exterior: 
                if distance <= neighbor_distance: # Make sure is the right end vertices (For exterior medial axis)
                    acc.append(candidate)
            else:
                acc.append(candidate)
    return acc


def plot_end_vertices(G, end_vertices, distance_map):
    main_fig, main_ax = plt.subplots(nrows=1, ncols=1, num='Layout', figsize=(12, 12))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)

    x_cord = [G.nodes[end_vertex]['x'] for end_vertex in end_vertices]
    y_cord = [distance_map.shape[0]-G.nodes[end_vertex]['y'] for end_vertex in end_vertices]
    plt.plot(x_cord, y_cord, marker='o', markersize=5, linewidth=0, color="red")

    plt.imshow(distance_map)
    
def debug_end_vertices(G, distance_map):
    main_fig, main_ax = plt.subplots(nrows=1, ncols=1, num='Layout', figsize=(12, 12))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    color = cm.get_cmap("Set3").colors
    i = 0
    nodes = list(G.nodes)
    for node in nodes:
        main_ax.plot(G.nodes[node]['x'], G.nodes[node]['y'], marker='o', markersize=5, linewidth=0)
        
def plot_boundary_vertices(resampled, width, height):
    main_fig, main_ax = plt.subplots(nrows=1, ncols=1, num='Layout', figsize=(width/100, height/100))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    line = list(resampled.coords)

    x = [point[0] for point in line]
    y = [point[1] for point in line]
    indices = [ind for ind, point in enumerate(x) if ind % 20 == 0]
    #indices_text = [str(ind) for ind in indices]
    main_ax.plot(x, y, marker='o', markersize=1, linewidth=.1)
    for i in indices:
        plt.text(x[i], y[i], str(i), fontsize=12)

from shapely.geometry import Point
import itertools
import math
import scipy

'''
Chech if two points' order is clockwise in the original polygon boundary
'''
def is_cw(linestring, index_1, index_2):
    coords = list(linestring.coords)
    total = len(coords)
    if index_1 < index_2 and (index_2 - index_1) < int(total / 2.0):
        return True
    elif index_1 < index_2 and (index_2 - index_1) > int(total / 2.0):
        return False
    elif index_1 > index_2 and (index_1 - index_2) > int(total / 2.0):
        return True
    else:
        return False

'''

return projection pairs [(x1, y1), (x2, y2)] in clockwise order
'''
def find_projection_pair(boundary_vertices, x, y, projection_distance, interior=True):
    single_point_projection_threshold = 8.0

    coords = list(boundary_vertices.coords)[:-1]
    length = len(coords)
    
    def vector_angle(vector_1, vector_2):
        vector_1 = vector_1 / np.linalg.norm(vector_1)
        vector_2 = vector_2 / np.linalg.norm(vector_2)

        ang = np.arccos(np.clip(np.dot(vector_1, vector_2), -1, 1))
        return ang

    kdTree = scipy.spatial.KDTree(np.array(coords))
    
    if projection_distance < single_point_projection_threshold: # too close to boundary
        d, nearest = kdTree.query([x, y], k=1)
        candidates = [(nearest-3)%length, (nearest+3)%length] 
        
        if is_cw(boundary_vertices, candidates[0], candidates[1]):
            return candidates
        else:
            reversed_list = candidates[::-1]
            return reversed_list
        
        return 
    
    neighbors = kdTree.query_ball_point([x, y], max(projection_distance*1.1, projection_distance+5.0))
    possible_projections = []
    possible_projections_backup = []
    
    # Check for orthogonality
    for neighbor in neighbors:
        vector1 = np.array(list(boundary_vertices.coords)[(neighbor+1)%length]) - np.array(list(boundary_vertices.coords)[neighbor])
        vector2 = np.array(list(boundary_vertices.coords)[neighbor]) - np.array([x, y])
        if abs(vector_angle(vector1, vector2) - 0.5 * math.pi) < math.pi / 18: # close to 90 degrees with 10 degrees of error
            possible_projections.append(neighbor)
        elif abs(vector_angle(vector1, vector2) - 0.5 * math.pi) < math.pi / 9: # close to 90 degrees with 20 degrees of error
            possible_projections_backup.append(neighbor)
    
    if len(possible_projections) < 2:
        possible_projections = possible_projections_backup
    
    if len(possible_projections) < 2:
        return possible_projections
    
    current_max = 0
    current_pair = []
    for projection_pair in itertools.combinations(possible_projections, 2):
        source = np.array([x, y])
        projection_1 = np.array(coords[projection_pair[0]])
        projection_2 = np.array(coords[projection_pair[1]])

        vector_1 = projection_1 - source
        vector_2 = projection_2 - source

        angle = vector_angle(vector_1, vector_2)
        if angle > current_max:
            current_max = angle
            current_pair = list(projection_pair)
            
    if is_cw(boundary_vertices, current_pair[0], current_pair[1]):
        return current_pair
    else:
        reversed_list = current_pair[::-1]
        return reversed_list
    
def plot_projection_pair(polygon, G, boundary_vertices, width, height):
    plot_polygon(polygon, width, height)
    for v in G.nodes:
        medial_vertex_x = G.nodes[v]['x']
        medial_vertex_y = G.nodes[v]['y']
        medial_vertex_distance = G.nodes[v]['distance']
        projections = find_projection_pair(boundary_vertices, medial_vertex_x, medial_vertex_y, medial_vertex_distance)
        for projection in projections:
            projection_x = list(boundary_vertices.coords)[projection][0]
            projection_y = list(boundary_vertices.coords)[projection][1]
            plt.plot([medial_vertex_x, projection_x], [medial_vertex_y, projection_y], marker='o', markersize=1, linewidth=1)
            
class BoundaryType(object):
    def __init__(self, index, vertex_index):
        self.index = index
        self.vertex_index = vertex_index


class NA(BoundaryType):
    def __repr__(self):
        return 'NA('+ str(self.index)+ ", vertex=" +  str(self.vertex_index) + ')'
    
    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, NA):
            return True
        return False

class Corner(BoundaryType):
    '''
    Index of the Corner
    '''
    def __init__(self, index, source, vertex_index, endpoints):
        self.index = index
        self.source = source
        self.vertex_index = vertex_index
        self.endpoints = endpoints
        
        # Calculate distance to corner center for future use
        if len(endpoints) ==  2:
            middle = int(round((endpoints[0] + endpoints[1]) / 2, 0))
            self.distance_to_center = abs(middle - vertex_index)
        else:
            self.distance_to_center = 0

    def __repr__(self):
        return 'Corner(' + str(self.index) + ", vertex=" +  str(self.vertex_index) + ')'
    
    def __eq__(self, other):
        if isinstance(other, Corner):
            return self.index == other.index
        return False

        
class Component(BoundaryType):
    def __repr__(self):
        return 'Component(' + str(self.index) + ')'
    
    def __eq__(self, other):
        if isinstance(other, Component):
            return self.index == other.index
        return False

from collections import OrderedDict
import matplotlib

def build_boundary_dic(boundary_vertices):
    od = OrderedDict()
    for vertex_index, point in list(enumerate(boundary_vertices.coords))[:-1]:
        od[point] = NA(-1, vertex_index)
    return od

def mark_corners(G, end_vertices, boundary_vertices_dict, boundary_vertices):
    corner_index = 0
    corner_mapping = {}
    for v in end_vertices:
        corner_mapping[v] = corner_index
        medial_vertex_x = G.nodes[v]['x']
        medial_vertex_y = G.nodes[v]['y']
        medial_vertex_distance = G.nodes[v]['distance']
        projections = find_projection_pair(boundary_vertices, medial_vertex_x, medial_vertex_y, medial_vertex_distance)
        coords = list(boundary_vertices.coords)[:-1]
        total = len(coords)
        
        if len(projections) == 2:
            if projections[1] > projections[0]:
                for i in range(projections[0], projections[1]+1):
                    vextex_index = boundary_vertices_dict[coords[i]].vertex_index
                    corner = Corner(corner_index, v, vextex_index, projections)
                    boundary_vertices_dict[coords[i]] = corner
            elif projections[1] < projections[0]:
                for i in range(projections[0], projections[1]+1+total):
                    vextex_index = boundary_vertices_dict[coords[i%total]].vertex_index
                    corner = Corner(corner_index, v, vextex_index, projections)
                    boundary_vertices_dict[coords[i%total]] = corner
        elif len(projections) == 1:
            vextex_index = boundary_vertices_dict[coords[projections[0]]].vertex_index
            corner = Corner(corner_index, v, vextex_index, projections)
            boundary_vertices_dict[coords[projections[0]]] = corner
        corner_index += 1
    return corner_mapping

def plot_corners(polygon, corner_dict, width, height, plot_component = False):
    # find corner indices
    corner_type = set()
    for k in corner_dict:
        if isinstance(corner_dict[k], Corner):
            corner_type.add(corner_dict[k].index)
    
    # set color to max corner index
    cmap = matplotlib.cm.get_cmap("jet", max(corner_type)+1)
    cmaplist = [cmap(i) for i in range(cmap.N)]
    plot_polygon(polygon, width, height)
    for k in corner_dict:
        if isinstance(corner_dict[k], Corner):
            plt.plot([k[0]], [k[1]], marker='o', markersize=4, linewidth=1, color=cmaplist[corner_dict[k].index])
        if plot_component:
            if isinstance(corner_dict[k], Component):
                plt.plot([k[0]], [k[1]], marker='x', markersize=4, linewidth=1, color=cmaplist[corner_dict[k].index])

def mark_component(boundary_vertices_dict):
    component_index = 0
    copied = boundary_vertices_dict.copy()
    
    filled = False
    # forward propagation
    for v in boundary_vertices_dict:
        if isinstance(boundary_vertices_dict[v], NA):
            vextex_index = boundary_vertices_dict[v].vertex_index
            copied[v] = Component(component_index, vextex_index)
            filled = True
        elif isinstance(boundary_vertices_dict[v], Corner) and filled == True:
            component_index += 1
            filled = False
            
    # back propagation
    if isinstance(copied[list(boundary_vertices_dict.keys())[0]], Component):
        i = -1
        while not isinstance(copied[list(boundary_vertices_dict.keys())[i]], Corner):
            if isinstance(copied[list(boundary_vertices_dict.keys())[i]], Component):
                copied[list(boundary_vertices_dict.keys())[i]] = copied[list(boundary_vertices_dict.keys())[0]]
            i = i - 1
            
    return copied

def mark_extended_corner(boundary_vertices_dict_ext, boundary_vertices_dict_int, boundary_vertices, G_ext, end_vertices_ext, corner_mapping_ext):
    extended_corner = boundary_vertices_dict_ext.copy()
    coords = list(boundary_vertices.coords)
    for end in end_vertices_ext:
        corner_index = corner_mapping_ext[end]
        concave_corner = Corner(corner_index, end, -1, [])

        for edges in nx.bfs_edges(G_ext, end):
            x = G_ext.nodes[edges[1]]['x']
            y = G_ext.nodes[edges[1]]['y']
            projection_distance = G_ext.nodes[edges[1]]['distance']
            projections = find_projection_pair(boundary_vertices, x, y, projection_distance)
            if len(projections) == 2:
                if (G_ext.degree(edges[1])<=2 and ((not isinstance(boundary_vertices_dict_int[coords[projections[0]]], Corner) and not isinstance(boundary_vertices_dict_int[coords[projections[1]]], Corner)))):
                    extended_corner[coords[projections[0]]] = concave_corner
                    extended_corner[coords[projections[1]]] = concave_corner
                else:
                    break
    return extended_corner

def adjust_corner(boundary_vertices_dict, gap = 10):
    
    def circular_any(start, end, collection, condition):
        length = len(collection)
        start = start
        end = end
        keyIndex = list(collection.keys())
        for i in range(start, end):
            if condition(collection[keyIndex[i % length]]):
                return collection[keyIndex[i % length]]
        return None

    copied = boundary_vertices_dict.copy()
    for i in range(len(boundary_vertices_dict)):
        left = circular_any(i-gap, i, boundary_vertices_dict, lambda x: isinstance(x, Corner))
        #print(i-10, i, left)
        right = circular_any(i+1, i+gap+1, boundary_vertices_dict, lambda x: isinstance(x, Corner))

        if left and right and left == right:
            key = list(boundary_vertices_dict.keys())[i]
            copied[key] = left
            #print("edit")
    return copied

def extract_projection_pair(G, boundary_vertices, return_source = False):
    coords = list(boundary_vertices.coords)
    projection_pairs = []
    sources = []
    for v in G.nodes:
        medial_vertex_x = G.nodes[v]['x']
        medial_vertex_y = G.nodes[v]['y']
        medial_vertex_distance = G.nodes[v]['distance']
        projections = find_projection_pair(boundary_vertices, medial_vertex_x, medial_vertex_y, medial_vertex_distance)
        if len(projections) == 2:
            projection_pairs.append((coords[projections[0]], coords[projections[1]]))
            sources.append(v)
    if return_source:
        return projection_pairs, sources
    else:
        return projection_pairs

def generate_raw_cuts(projection_pairs, corner_dict):
    raw_cuts = []
    for pp in projection_pairs:
        if isinstance(corner_dict[pp[0]], Corner) or isinstance(corner_dict[pp[1]], Corner):
            # Remove potential duplicated
            if pp not in raw_cuts:
                raw_cuts.append(pp)
    return raw_cuts

def plot_raw_cuts(raw_cuts, polygon, width, height):
    plot_polygon(polygon, width, height)
    for raw_cut in raw_cuts:
        plt.plot([raw_cut[0][0], raw_cut[1][0]], [raw_cut[0][1], raw_cut[1][1]], color=(0.76, 0.99, 0.29, 0.5))
        
def strong_equivalence(pair_1_coords, pair_2_coords, boundary_vertices_dict_ext, boundary_vertices_dict_int):
    pair_1 = (boundary_vertices_dict_int[pair_1_coords[0]], boundary_vertices_dict_int[pair_1_coords[1]])
    pair_2 = (boundary_vertices_dict_int[pair_2_coords[0]], boundary_vertices_dict_int[pair_2_coords[1]])
    same_branch = False
    if pair_1 == pair_2:
        same_branch = True
    elif pair_2[0] == pair_1[1] and pair_2[1] == pair_1[0]:
        same_branch = True
    else:
        same_branch = False
        
    pair_1 = (boundary_vertices_dict_ext[pair_1_coords[0]], boundary_vertices_dict_ext[pair_1_coords[1]])
    pair_2 = (boundary_vertices_dict_ext[pair_2_coords[0]], boundary_vertices_dict_ext[pair_2_coords[1]])
    same_corner = False
    if pair_1 == pair_2:
        same_corner = True
    elif pair_2[0] == pair_1[1] and pair_2[1] == pair_1[0]:
        same_corner = True
    else:
        same_corner = False
    
    return same_branch and same_corner

def cut_distance_to_center(cut):
    if isinstance(cut[0], Corner) and not isinstance(cut[1], Corner):
        return cut[0].distance_to_center
    elif not isinstance(cut[0], Corner) and isinstance(cut[1], Corner):
        return cut[1].distance_to_center
    elif isinstance(cut[0], Corner) and isinstance(cut[1], Corner):
        return cut[0].distance_to_center + cut[1].distance_to_center
    else:
        return -1

def cut2boundary_group(cut, boundary_vertices_dict):
    return (boundary_vertices_dict[cut[0]], boundary_vertices_dict[cut[1]])
    

from operator import itemgetter
def select_representative_cuts(raw_cuts, boundary_vertices_dict_ext, boundary_vertices_dict_int):
    copied = raw_cuts.copy()

    representative_cuts = []
    while copied:
        target = copied.pop()
        equivalence_set = [target]
        for remains in copied:
            if strong_equivalence(target, remains, boundary_vertices_dict_ext, boundary_vertices_dict_int):
                equivalence_set.append(remains)
        distance = [cut_distance_to_center(cut2boundary_group(element, boundary_vertices_dict_ext)) for element in equivalence_set]
        index, element = min(enumerate(distance), key=itemgetter(1))
        representative_cuts.append(equivalence_set[index])

        for element in equivalence_set[1:]:
            copied.remove(element)
    return representative_cuts

def protrusion_strength(vertex_1, vertex_2, boundary_linestring):
    euclidean_distance = math.sqrt((vertex_1[0] - vertex_2[0]) ** 2 + (vertex_1[1] - vertex_2[1]) ** 2)
    min_arc = minimal_arc(vertex_1, vertex_2, boundary_linestring)
    
    return euclidean_distance / min_arc

def minimal_arc(vertex_1, vertex_2, boundary_linestring):
    total_length = boundary_linestring.length
    projection_1 = boundary_linestring.project(Point(vertex_1))
    projection_2 = boundary_linestring.project(Point(vertex_2))
    
    arc_length = abs(projection_2 - projection_1)
    return min(arc_length, total_length - arc_length) # Select the minimal arc length

def plot_cuts_score(cuts, text, polygon, width, height):
    plot_raw_cuts(cuts, polygon, width, height)
    mid_point = [((c[0][0] + c[1][0])/2, (c[0][1] + c[1][1])/2) for c in cuts]
    i = 0 
    for x, y in mid_point:
        plt.text(x, y, text[i], fontsize=12)
        i = i + 1

def get_extented_arc_length(boundary_vertices_dict_extended, distance=5):
    length_dict = {}
    for v in boundary_vertices_dict_extended:
        if isinstance(boundary_vertices_dict_extended[v], Corner) and boundary_vertices_dict_extended[v].index in length_dict:
            length_dict[boundary_vertices_dict_extended[v].index] = length_dict[boundary_vertices_dict_extended[v].index] + distance
        elif isinstance(boundary_vertices_dict_extended[v], Corner) and not(boundary_vertices_dict_extended[v].index in length_dict):
            length_dict[boundary_vertices_dict_extended[v].index] = distance
    return length_dict

def extension_strength(cut, extended_arc_length, distance_adjusted_ext):
    if isinstance(distance_adjusted_ext[cut[0]], Corner) and isinstance(distance_adjusted_ext[cut[1]], Corner):
        es = (extended_arc_length.get(distance_adjusted_ext[cut[0]].index, 0) + extended_arc_length.get(distance_adjusted_ext[cut[1]].index, 0))/2
    elif isinstance(distance_adjusted_ext[cut[0]], Corner) and not isinstance(distance_adjusted_ext[cut[1]], Corner):
        es = float(extended_arc_length.get(distance_adjusted_ext[cut[0]].index, 0))
    elif not isinstance(distance_adjusted_ext[cut[0]], Corner) and isinstance(distance_adjusted_ext[cut[1]], Corner):
        es = float(extended_arc_length.get(distance_adjusted_ext[cut[1]].index, 0))
        
    euclidean_distance = math.sqrt((cut[0][0] - cut[1][0]) ** 2 + (cut[0][1] - cut[1][1]) ** 2)
    
    return es / euclidean_distance

def isDouble(cuts, boundary_vertices_dict):
    if isinstance(boundary_vertices_dict[cuts[0]], Corner) and isinstance(boundary_vertices_dict[cuts[1]], Corner):
        return True
    else:
        return False

def extract_corner_info(boundary_vertices_dict):
    corner_source = {}
    corner_endpoints = {}
    for key in boundary_vertices_dict:
        if isinstance(boundary_vertices_dict[key], Corner):
            corner_index = boundary_vertices_dict[key].index
            corner_source[corner_index] = boundary_vertices_dict[key].source
            corner_endpoints[corner_index] = boundary_vertices_dict[key].endpoints
    return corner_source, corner_endpoints

def vector_angle(vector_1, vector_2):
    vector_1 = vector_1 / np.linalg.norm(vector_1)
    vector_2 = vector_2 / np.linalg.norm(vector_2)

    ang = np.arccos(np.clip(np.dot(vector_1, vector_2), -1, 1))
    return ang

def plot_corner_endpoints(corner_mappings, corner_endpoints, boundary_vertices, G):
    coords = list(boundary_vertices.coords)
    for endv in corner_mappings:
        x = G.nodes[endv]['x']
        y = G.nodes[endv]['y']
        plt.plot([x], [y], marker='o', markersize=6, color='red')
        vectors = []

        # If initial endpoint not in current boundary vertex dict
        if corner_mappings[endv] in corner_endpoints:
            for projection in corner_endpoints[corner_mappings[endv]]:
                vectors.append((coords[projection][0] - x, coords[projection][1] - y))
                plt.plot([x, coords[projection][0]], [y, coords[projection][1]], marker='o', markersize=1, color='blue', linewidth=1)
            try:
                tt = "{va:.2f}".format(va=vector_angle(vectors[0], vectors[1])/math.pi*180.0)
                plt.text(x, y, tt, fontsize=12)
            except:
                pass

'''
Vectors are given in counter-clockwise order.
'''
def interior_angle(vector_1, vector_2):
    vector_1 = vector_1 / np.linalg.norm(vector_1)
    vector_2 = vector_2 / np.linalg.norm(vector_2)

    ang = np.arccos(np.clip(np.dot(vector_1, vector_2), -1, 1))
    ang = np.abs(ang) if np.cross(vector_1, vector_2) > 0 else 2*np.pi - np.abs(ang)
    return ang

def create_unit_vector(raw_vector):
    vector = np.array(raw_vector)
    v = vector / np.linalg.norm(vector)
    return v

'''
points given in clockwise order is assumed
'''

def point_tangent_vector(boundary_vertices, vertex_index, is_first):
    coords = list(boundary_vertices.coords)
    length = len(coords)
    vector_tail = np.array(coords[vertex_index])
    if is_first:
        vector_head = np.array(coords[(vertex_index-2)%length])
    else:
        vector_head = np.array(coords[(vertex_index+2)%length])
    return create_unit_vector(vector_head - vector_tail)

def corner_calibrated_endpoints(corner_mappings, corner_endpoints, boundary_vertices, G, expansion_parameter=5):
    coords = list(boundary_vertices.coords)
    total = len(coords)
    calibrated = {}
    for corner in corner_endpoints:
        projections = corner_endpoints[corner]
        if len(projections) == 2:
            calibrated[corner] = projections
        if len(projections) == 1:
            calibrated[corner] = [(projections[0]-expansion_parameter)%total, (projections[0]+expansion_parameter)%total]
    return calibrated

def plot_corner_interior_angle(corner_mappings, corner_endpoints, boundary_vertices, G, expansion_parameter=5):
    coords = list(boundary_vertices.coords)
    total = len(coords)
    for endv in corner_mappings:
        x = G.nodes[endv]['x']
        y = G.nodes[endv]['y']
        
        # If initial endpoint not in current boundary vertex dict
        if corner_mappings[endv] in corner_endpoints:
            projections = corner_endpoints[corner_mappings[endv]]
            if len(projections) == 2:
                v1 = point_tangent_vector(boundary_vertices, projections[0], is_first=True)
                v2 = point_tangent_vector(boundary_vertices, projections[1], is_first=False)
                angle = interior_angle(v1, v2)
                tt = "{a:.2f}".format(a=angle*180/math.pi)
                plt.text(x, y, tt, fontsize=12)
            elif len(projections) == 1:
                v1 = point_tangent_vector(boundary_vertices, (projections[0]-expansion_parameter)%total, is_first=True)
                v2 = point_tangent_vector(boundary_vertices, (projections[0]+expansion_parameter)%total, is_first=False)
                angle = interior_angle(v1, v2)
                tt = "{a:.2f}".format(a=angle*180/math.pi)
                plt.text(x, y, tt, fontsize=12)
            
import math
'''
start -> end follows counter-clockwise order
'''
class Node:
    def __init__(self, start_vector, end_vector):
        self.start_vector = start_vector
        self.end_vector = end_vector
        #self.cut_vector = None
        self.left_child = None #smaller
        self.right_child = None #greater
    
    def is_leave(self):
        if not self.left_child or not self.right_child:
            return True
        else:
            return False
    
    def contains(self, cut_vector):
        ordered_vector = [self.start_vector, cut_vector, self.end_vector]
        ring = LinearRing(ordered_vector)
        if ring.is_ccw and ring.is_valid:
            return True
        else:
            return False
    
    '''
    Angle in radian, only display in degree
    '''
    def angle(self):
        return interior_angle(self.start_vector, self.end_vector)
        
    def __repr__(self):
        return 'Node(' + str(self.start_vector) + ", " +  str(self.end_vector) + ", angle=" + str(self.angle()*180/math.pi) +')'
        
class InteriorAngle:
    def __init__(self, start_vector, end_vector):
        self.root = Node(np.array(start_vector), np.array(end_vector))
        self.tolerance = math.pi / 12.0
        self.cut_list = []
        
    def add_cut(self, cut_vector):
        # keep track of all the cuts
        if self.root.contains(cut_vector):
            self.cut_list.append(create_unit_vector(cut_vector))
        self._add_cut(create_unit_vector(cut_vector), self.root)
        
            
    def _add_cut(self, cut_vector, cur_node):
        if cur_node.contains(cut_vector):
            if cur_node.left_child and cur_node.right_child:
                if cur_node.left_child.contains(cut_vector):
                    self._add_cut(cut_vector, cur_node.left_child)
                else:
                    self._add_cut(cut_vector, cur_node.right_child)
            else:
                cur_node.left_child = Node(cur_node.start_vector, cut_vector)
                cur_node.right_child = Node(cut_vector, cur_node.end_vector)
        else:
            print("Invalid cut")
    
    def is_convex(self):
        angles = self.list_leaves()
        return all(angle.angle() < math.pi + self.tolerance for angle in angles)
    
    def list_leaves(self):
        return self._list_leaves(self.root)
        
    def _list_leaves(self, cur_node):
        if cur_node.is_leave():
            return [cur_node]
        else:
            return self._list_leaves(cur_node.left_child) + self._list_leaves(cur_node.right_child)
        
    def print_tree(self):
        print(self._print_tree(self.root))

    def _print_tree(self,cur_node):
        if cur_node.is_leave():
            return str(cur_node)
        else:
            return '[' + self._print_tree(cur_node.left_child) + ", "+ self._print_tree(cur_node.right_child) + "]"

def cut_corner_mappings(cuts, boundary_vertices_dict):
    cut_2_corner = {}
    corner_2_cut = {}
    for cut in cuts:
        acc_corner = []

        label1 = boundary_vertices_dict[cut[0]]
        label2 = boundary_vertices_dict[cut[1]]
        if isinstance(label1, Corner):
            acc_corner.append(label1.index)
            if label1.index not in corner_2_cut:
                corner_2_cut[label1.index] = [cut]
            elif label1.index in corner_2_cut:
                corner_2_cut[label1.index].append(cut)
        if isinstance(label2, Corner):
            acc_corner.append(label2.index)
            if label2.index not in corner_2_cut:
                corner_2_cut[label2.index] = [cut]
            elif label2.index in corner_2_cut:
                corner_2_cut[label2.index].append(cut)
        cut_2_corner[cut] = acc_corner
    return cut_2_corner, corner_2_cut

def calculate_corner_residue(corner2rawcuts, boundary_vertices):
    corner_residue = {}
    for key in corner2rawcuts:
        residue = [minimal_arc(cut[0], cut[1], boundary_vertices) for cut in corner2rawcuts[key]]
        max_residue = np.max(residue)
        corner_residue[key] = max_residue

    return corner_residue

def plot_corner_residue(corner_mappings, corner_endpoints, corner_residue, G):
    for endv in corner_mappings:
        x = G.nodes[endv]['x']
        y = G.nodes[endv]['y']
        # Check if corner has cuts
        if corner_mappings[endv] in corner_residue:
            tt = "{a:.2f}".format(a=corner_residue[corner_mappings[endv]])
            plt.text(x, y, tt, fontsize=12)

def create_cut_vector(cut, corner_index, boundary_vertices_dict):
    label1 = boundary_vertices_dict[cut[0]]
    label2 = boundary_vertices_dict[cut[1]]
    if isinstance(label1, Corner) and label1.index == corner_index:
        tail = np.array(cut[0])
        head = np.array(cut[1])
        cut_vector = create_unit_vector(head - tail)
    elif isinstance(label2, Corner) and label2.index == corner_index:
        tail = np.array(cut[1])
        head = np.array(cut[0])
        cut_vector = create_unit_vector(head - tail)
    return cut_vector

def generate_cuts(filename, output_path):
    
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if not path.exists(output_path):
        os.mkdir(output_path)
    
    polygon = generate_canvas_polygon(image)[0]
    medial_exterior_input = prepare_for_medial_axis(image, complement=True)
    medial_interior_input = prepare_for_medial_axis(image, complement=False)

    ma_ext = ridge_medial_axis(medial_exterior_input, ridge_threshold = 0.3, small_threshold=5)
    ma_int = ridge_medial_axis(medial_interior_input, ridge_threshold = 0.3, small_threshold=5)

    # Check if there is concave corner. If not, no need for decomposition
    if ma_ext[0].sum() > 0:
    
        # exterior medial axis graph
        multilinestring_ext = build_medial_multilinestring(ma_ext[0])
        final_medial_vertices_ext = redistribute_vertices(multilinestring_ext[0], 5)
        G_ext = build_medial_graph(final_medial_vertices_ext, multilinestring_ext[1], ma_ext[1])
        endv_ext = find_end_vertices(G_ext, exterior=True)


        # interior medial axis graph
        multilinestring_int = build_medial_multilinestring(ma_int[0])
        final_medial_vertices_int = redistribute_vertices(multilinestring_int[0], 5)
        G_int = build_medial_graph(final_medial_vertices_int, multilinestring_int[1], ma_int[1])
        endv_int = find_end_vertices(G_int, exterior=False)


        boundary_vertices = redistribute_vertices(LineString(polygon.exterior.coords), 5)
        boundary_vertices_dict_ext = build_boundary_dic(boundary_vertices)
        corner_mapping_ext = mark_corners(G_ext, endv_ext, boundary_vertices_dict_ext, boundary_vertices)
        component_adjusted_ext = mark_component(boundary_vertices_dict_ext)

        boundary_vertices_dict_int = build_boundary_dic(boundary_vertices)
        corner_mapping_int = mark_corners(G_int, endv_int, boundary_vertices_dict_int, boundary_vertices)
        component_adjusted_int = mark_component(boundary_vertices_dict_int)


        ec = mark_extended_corner(component_adjusted_ext, component_adjusted_int, boundary_vertices, G_ext, endv_ext, corner_mapping_ext)
        ec_adjusted = adjust_corner(ec)

        projection_pairs = extract_projection_pair(G_int, boundary_vertices)
        raw_cuts = generate_raw_cuts(projection_pairs, component_adjusted_ext)

        plot_raw_cuts(raw_cuts, polygon, image.shape[1], image.shape[0])
        plt.savefig(join(output_path, 'raw.png'), bbox_inches='tight')
        plt.clf()

        # representative
        representative = select_representative_cuts(raw_cuts, component_adjusted_ext, component_adjusted_int)
        plot_raw_cuts(representative, polygon, image.shape[1], image.shape[0])
        plt.savefig(join(output_path, 'representitive.png'), bbox_inches='tight')
        plt.clf()

        # plot corners
        plot_corners(polygon, component_adjusted_ext, image.shape[1], image.shape[0], plot_component = False)
        plt.savefig(join(output_path, 'corner.png') , bbox_inches='tight')
        plt.clf()

        # add a little filter
        denoised = [r for r in representative if protrusion_strength(r[0], r[1], boundary_vertices) < 0.75]
        text = ["{ps:.2f}".format(ps = protrusion_strength(r[0], r[1], boundary_vertices)) for r in denoised]
        plot_cuts_score(denoised, text, polygon, image.shape[1], image.shape[0])
        plt.savefig(join(output_path, 'protrusion.png'), bbox_inches='tight')
        plt.clf()

        # Extension strength
        plot_corners(polygon, ec_adjusted, image.shape[1], image.shape[0], plot_component = False)
        plt.savefig(join(output_path,'extended_corner.png'), bbox_inches='tight')
        plt.clf()

        # Extendted arc length
        eal = get_extented_arc_length(ec_adjusted, distance=5)
        text = ["{es:.2f}".format(es=extension_strength(r, eal, component_adjusted_ext)) for r in representative]
        plot_cuts_score(representative, text, polygon, image.shape[1], image.shape[0])
        plt.savefig(join(output_path, 'extension.png'), bbox_inches='tight')
        plt.clf()

        # Local Convexity
        priority = np.array([1 if isDouble(r, component_adjusted_ext) else 0 for r in denoised])
        text = ["Double" if r else "Single" for r in priority]
        plot_cuts_score(denoised, text, polygon, image.shape[1], image.shape[0])
        plt.savefig(join(output_path, 'single_double.png'), bbox_inches='tight')
        plt.clf()

        eal = get_extented_arc_length(ec_adjusted, distance=5)
        protrusion_threshold = 0.5
        extension_strength_threshold = 0.9

        protrusion_strength_filter = np.array([protrusion_strength(r[0], r[1], boundary_vertices) > protrusion_threshold for r in denoised]) # priority decreased
        extension_strength_filter = np.array([extension_strength(r, eal, component_adjusted_ext) < extension_strength_threshold for r in denoised])  # priority decreased

        priority_decrease = np.logical_or(protrusion_strength_filter, extension_strength_filter).astype(int)
        saliency_adjusted_priority = priority - priority_decrease

        corner_source, corner_endpoints = extract_corner_info(component_adjusted_ext)
        calibrated = corner_calibrated_endpoints(corner_mapping_ext, corner_endpoints, boundary_vertices, G_ext)

        plot_raw_cuts(representative, polygon, image.shape[1], image.shape[0])
        plot_corner_interior_angle(corner_mapping_ext, corner_endpoints, boundary_vertices, G_ext)
        plt.savefig(join(output_path, 'interior_angle.png'), bbox_inches='tight')
        plt.clf()
        cut_2_corner, corner_2_cuts = cut_corner_mappings(denoised, component_adjusted_ext)

        # residue for a corner
        _, corner2rawcuts = cut_corner_mappings(raw_cuts, component_adjusted_ext)

        # corner chord residue calculation
        corner_residue = calculate_corner_residue(corner2rawcuts, boundary_vertices)
        plot_raw_cuts(denoised, polygon, image.shape[1], image.shape[0])
        plot_corner_residue(corner_mapping_ext, corner_endpoints, corner_residue, G_ext)
        plt.savefig(join(output_path, 'corner_residue.png'), bbox_inches='tight')
        plt.clf()

        # Sort the corner based on chord residue
        corner_ordered = [corner for corner, _ in sorted(corner_residue.items(), key=lambda item: item[1])]

        '''
        Interatively adding cuts until corners become convex or all the cuts have been depleted.
        '''
        angle_equivalence_threshold = math.pi / 20.0 # threshold for cuts that are consider similar
        tolerance = math.pi / 12.0 # threshold for convexity
        final_cuts = []
        for corner in corner_ordered:
            #print("Now Processing Corner: ", corner)
            if corner in corner_2_cuts:
                cuts = corner_2_cuts[corner].copy()

                cuts.sort(key=lambda x:saliency_adjusted_priority[denoised.index(x)])
                # Build a Concave Corner
                start_vector = point_tangent_vector(boundary_vertices, calibrated[corner][0], is_first=True)
                end_vector = point_tangent_vector(boundary_vertices, calibrated[corner][1], is_first=False)
                concave_corner = InteriorAngle(start_vector, end_vector)
                #concave_corner.print_tree()

                # Add double cuts that are already selected
                for cut in cuts:
                    if cut in final_cuts:
                        cut_vector = create_cut_vector(cut, corner, component_adjusted_ext)
                        concave_corner.add_cut(cut_vector)

                # Skip corners that is already convex
                if concave_corner.is_convex():
                    continue

                # Add cuts until all cuts has been added
                while cuts:
                    current_cut = cuts.pop()
                    if current_cut in final_cuts:
                        continue

                    cut_vector = create_cut_vector(current_cut, corner, component_adjusted_ext)

                    # Check current cut is different than cuts that is already selected
                    difference = [interior_angle(cut_vector, past_cut_angle) for past_cut_angle in concave_corner.cut_list]
                    if all(d > angle_equivalence_threshold for d in difference):
                        concave_corner.add_cut(cut_vector)
                        if current_cut not in final_cuts:
                            final_cuts.append(current_cut)
                        # Check convexity
                        angles = [leaf.angle() for leaf in concave_corner.list_leaves()]#angles_between(acc, start_vector, end_vector)
                        if all(a < math.pi + tolerance for a in angles):
                            break
        plot_raw_cuts(final_cuts, polygon, image.shape[1], image.shape[0])
        plt.savefig(join(output_path, 'final.png'), bbox_inches='tight')
        plt.clf()
        with open(join(output_path, 'final_cut.json'), 'w') as f:
            json.dump(final_cuts, f)
    else:
        print("No need for decomposition. Already convex polygon")
        with open(join(output_path, 'final_cut.json'), 'w') as f:
            json.dump([], f)
        plot_polygon(polygon, image.shape[1], image.shape[0])
        plt.savefig(join(output_path, 'raw.png'), bbox_inches='tight')
        plt.clf()

        

if __name__ == '__main__':
    generate_cuts(sys.argv[1], sys.argv[2])