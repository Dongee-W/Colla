from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
import numpy as np
import cv2
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import sys
import json
import seam_carving

def load_color_image(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    alpha_channel_added = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    return alpha_channel_added

'''
Scale image if too large (The preprocessing step)
'''
def preprocess_image(img):
    max_side = max(img.shape[0], img.shape[1])
    if max_side > 1500:
        scale_factor = 1500 / max_side
        img = cv2.resize(img, (int(img.shape[1]*scale_factor), int(img.shape[0]*scale_factor)))
    return img


def write_color_image(array, path):
    bgr = cv2.cvtColor(array, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(path, bgr)
    
def rowcol2xy(row, col, ymax):
    return int(col), int(ymax - row)

def xy2rowcol(x, y , ymax):
    return int(round(ymax - y, 0)), int(round(x, 0))

def retarget(image, width, height):
    return cv2.resize(image, (width, height))
'''
move origin to minX, minY of the polygon bounding box
'''
def polygon2local_coordinate(polygon):
    bounding_box = polygon.bounds
    return np.array([(int(coord[0] - bounding_box[0]), int(coord[1] - bounding_box[1]))
                     for coord in list(polygon.exterior.coords)])

'''
overaly an image over the target image at origin (in target image coordinate)
origin: (starting row, starting column)
'''
def image_overlay(target, source, origin):
    target = target.copy()
    source_crop = source.copy()
    # Case 1:
    if origin[0]<0 and origin[0] + source.shape[0] -1 <= target.shape[0]:
        start_row = 0
        end_row = origin[0] + source.shape[0]
        source_crop = source_crop[-origin[0]:,:].copy()
    # Case 2:
    elif origin[0]>=0 and origin[0] + source.shape[0] -1 <= target.shape[0]:
        start_row = origin[0]
        end_row = origin[0] + source.shape[0]
        source_crop = source_crop.copy()
    # Case 3
    elif origin[0]>=0 and origin[0] + source.shape[0] -1 > target.shape[0]:
        start_row = origin[0]
        end_row = target.shape[0]-1
        source_crop = source_crop[0:target.shape[0]-origin[0]-1,:].copy()
    # Case 4
    else:
        start_row = 0
        end_row = target.shape[0]-1
        source_crop = source_crop[-origin[0]:target.shape[0]-origin[0]-1,:].copy()
    
    # Case 1:
    if origin[1]<0 and origin[1] + source.shape[1] -1 <= target.shape[1]:
        start_col = 0
        end_col = origin[1] + source.shape[1]
        source_crop = source_crop[:,-origin[1]:].copy()
    # Case 2:
    elif origin[1]>=0 and origin[1] + source.shape[1] -1 <= target.shape[1]:
        start_col = origin[1]
        end_col = origin[1] + source.shape[1]
        source_crop = source_crop.copy()
    # Case 3
    elif origin[1]>=0 and origin[1] + source.shape[1] -1 > target.shape[1]:
        start_col = origin[1]
        end_col = target.shape[1]-1
        source_crop = source_crop[:,0:target.shape[1]-origin[1]-1].copy()
    # Case 4
    else:
        start_col = 0
        end_col = target.shape[1]-1
        source_crop = source_crop[:,-origin[1]:target.shape[1]-origin[1]-1].copy()
    
    target[start_row:end_row, start_col:end_col] = source_crop
    return target

'''
enlarge the main object rectangle to add some margin
if touch to boundary, return True
input format: (x1, x2, y1, y2)
'''
def adjust_inner_rec(outer, inner):
    outer_width = outer[1] - outer[0]
    outer_height = outer[3] - outer[2]
    
    inner_width = inner[1] - inner[0]
    inner_height = inner[3] - inner[2]
    margin_width = int(inner_width/18)
    margin_height = int(inner_height/18)
    
    new_x1 = max(inner[0]-margin_width, int(outer_width/120))
    new_x2 = min(inner[1]+margin_width, outer[1]-int(outer_width/120))
    new_y1 = max(inner[2]-margin_height, int(outer_height/120))
    new_y2 = min(inner[3]+margin_height, outer[3]-int(outer_height/120))
    
#     new_x1 = max(inner[0]-margin_width, 0)
#     new_x2 = min(inner[1]+margin_width, outer[1])
#     new_y1 = max(inner[2]-margin_height, 0)
#     new_y2 = min(inner[3]+margin_height, outer[3])
    
    touch_boundary = False
    if new_x1==0 or new_x2==outer[1] or new_y1 == 0 or new_y2==outer[3]:
        touch_boundary = True
    
    return (new_x1, new_x2, new_y1, new_y2), touch_boundary

'''
Get the triangulation given outer and innter rectangles (counter-clockwise order start from (0,0))
    [
        bottem left, bottom right, top right, top left
    ]
'''
def triangulation(outer_rec, inner_rec, height):
    triangles = [[outer_rec[3], inner_rec[3], outer_rec[2]],
     [inner_rec[3], inner_rec[2], outer_rec[2]],
     [inner_rec[2], outer_rec[1], outer_rec[2]],
     [inner_rec[2], inner_rec[1], outer_rec[1]],
     [inner_rec[0], outer_rec[1], inner_rec[1]],
     [outer_rec[0], outer_rec[1], inner_rec[0]],
     [outer_rec[0], inner_rec[0], inner_rec[3]],
     [outer_rec[3], outer_rec[0], inner_rec[3]],
     [inner_rec[3], inner_rec[2], inner_rec[1]],
     [inner_rec[0], inner_rec[1], inner_rec[3]]
    ]
    return [[(vertex[0], height-vertex[1]) for vertex in t] for t in triangles]

'''
masks are uint8 array of shape (height, width)
'''
def overlay_mask(mask1, mask2):
    overlaps = cv2.bitwise_and(mask1, mask2)
    return mask2 - overlaps # remove overlaps
    

def retarget_warp(image, 
             outer_rectangle_source,
             inner_rectangle_source,
             outer_rectangle_dest,
             inner_rectangle_dest
            ):
    width, height = outer_rectangle_dest[2]

    src_triangulation = triangulation(outer_rectangle_source, inner_rectangle_source, image.shape[0])
    dest_triangulation = triangulation(outer_rectangle_dest, inner_rectangle_dest, height)

    whole_canvas = np.zeros((height, width, 4), dtype=np.uint8)
    whole_mask = np.zeros((height, width), dtype=np.uint8)
    for idx in range(len(src_triangulation)):
        warp_mat = cv2.getAffineTransform(np.array(src_triangulation[idx]).astype(np.float32), np.array(dest_triangulation[idx]).astype(np.float32))
        warp_dst = cv2.warpAffine(image.copy(), warp_mat, (width, height),cv2.INTER_NEAREST)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(mask, [np.array(dest_triangulation[idx]).astype(np.int32)], 0, 255, -1).astype(np.uint8)
        new_mask = overlay_mask(whole_mask, mask) # accumulate mask for avoiding overlapping
        patch = cv2.bitwise_and(warp_dst, warp_dst, mask = new_mask)
        whole_canvas += patch
        whole_mask += new_mask
    
    return whole_canvas

def retarget_seam_carving(image, target_width, target_height):
    scale_factor = max(target_width/image.shape[1], target_height/image.shape[0])
    scaled = cv2.resize(image, (int(image.shape[1]*scale_factor), int(image.shape[0]*scale_factor)))
    dst = seam_carving.resize(
        scaled[:,:,0:3], (target_width, target_height),
        energy_mode='backward',   # Choose from {backward, forward}
        order='height-first',  # Choose from {width-first, height-first}
        keep_mask=None
    )
    alpha_channel = np.zeros((dst.shape[0], dst.shape[1], 1), dtype=np.uint8)+255
    new_dst = np.concatenate([dst, alpha_channel], axis=2)
    return new_dst
    
    

'''
part: partition dict of the format
    {'index': 12,
    'coords': [[796.0365929472149, 609.0],.....],
    'foreground': [x1,x2,y1,y2]}
 
image: image dict of the format
    {'filename': '02.jpg', 'foreground': [315, 700, 1, 1043], 'assigned_part': 12}
'''
from shapely.affinity import scale

def generate_image_patch(part_dict, image_dict, image_directory, whole_canvas_height, magnification=1.0):
    #part = scale(part, xfact=magnification, yfact=magnification, origin=(0,0))
    
    polygon = Polygon(part_dict['coords'])
    polygon_scaled = scale(polygon, xfact=magnification, yfact=magnification, origin=(0,0))
    
    bounding_box = polygon_scaled.bounds
    polygon_space_origin = bounding_box[0], bounding_box[1]
    width = int(bounding_box[2] - bounding_box[0])+1
    height = int(bounding_box[3] - bounding_box[1])+1
        
    # Load images
    image = load_color_image(join(image_directory, image_dict["filename"]))
    # enlarge the inner rectangle to add margins around main object
    enlarged_inner, touch_boundary = adjust_inner_rec([0, image.shape[1], 0, image.shape[0]], image_dict['foreground'])
    
    touch_boundary = False

    if image_dict['foreground_exists'] and not touch_boundary:       
        outer_rectangle_source = [(0,0), (image.shape[1], 0), (image.shape[1], image.shape[0]), (0, image.shape[0])]
        inner_rectangle_source = [(enlarged_inner[0], enlarged_inner[2]), 
                           (enlarged_inner[1], enlarged_inner[2]),
                           (enlarged_inner[1], enlarged_inner[3]),
                           (enlarged_inner[0], enlarged_inner[3])]

        new_x1 = part_dict['foreground'][0]*magnification - polygon_space_origin[0]
        new_x2 = part_dict['foreground'][1]*magnification - polygon_space_origin[0]
        new_y1 = part_dict['foreground'][2]*magnification - polygon_space_origin[1]
        new_y2 = part_dict['foreground'][3]*magnification - polygon_space_origin[1]

        outer_rectangle_dest = [(0,0), (width, 0), (width, height), (0, height)]
        inner_rectangle_dest = [(new_x1, new_y1), 
                           (new_x2, new_y1),
                           (new_x2, new_y2),
                           (new_x1, new_y2)]
        
        # Crop the image to the size of part proportionally
        x1_diff_source = enlarged_inner[0] # distance to left outer to left inner
        x2_diff_source = image.shape[1] - enlarged_inner[1]
        y1_diff_source = enlarged_inner[2]
        y2_diff_source = image.shape[0] - enlarged_inner[3]

        inner_width_source = enlarged_inner[1] - enlarged_inner[0]
        inner_height_source = enlarged_inner[3] - enlarged_inner[2]


        x1_diff_dest = new_x1
        x2_diff_dest = width - new_x2
        y1_diff_dest = new_y1
        y2_diff_dest = height - new_y2

        inner_width_dest = new_y2 - new_y1
        inner_height_dest = new_x2 - new_x1

        new_x1_outer_source = 0
        new_x2_outer_source = image.shape[1]
        new_y1_outer_source = 0
        new_y2_outer_source = image.shape[0]


        if x1_diff_dest / inner_width_dest < x1_diff_source/inner_width_source:
            new_x1_outer_source = enlarged_inner[0] - x1_diff_dest / inner_width_dest * inner_width_source

        if x2_diff_dest / inner_width_dest < x2_diff_source/inner_width_source:
            new_x2_outer_source = enlarged_inner[1] + x2_diff_dest / inner_width_dest * inner_width_source

        if y1_diff_dest / inner_height_dest < y1_diff_source/inner_height_source:
            new_y1_outer_source = enlarged_inner[2] - y1_diff_dest / inner_height_dest * inner_height_source

        if y2_diff_dest / inner_height_dest < y2_diff_source/inner_height_source:
            new_y2_outer_source = enlarged_inner[3] + y2_diff_dest / inner_height_dest * inner_height_source
            
        # Crop the image first end

        new_outer_source = [int(new_x1_outer_source), int(new_x2_outer_source), int(new_y1_outer_source), int(new_y2_outer_source)]
        new_outer_rectangle_source = [(new_outer_source[0], new_outer_source[2]), 
                           (new_outer_source[1], new_outer_source[2]),
                           (new_outer_source[1], new_outer_source[3]),
                           (new_outer_source[0], new_outer_source[3])]
        retargeted = retarget_warp(image, new_outer_rectangle_source, inner_rectangle_source, outer_rectangle_dest, inner_rectangle_dest)


    else:
        retargeted = retarget_seam_carving(image, width, height)

    polygon_mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(polygon_mask, [polygon2local_coordinate(polygon_scaled)], (255))
    
    #if magnification > 1:
    blur = cv2.GaussianBlur(polygon_mask,(7,7),0)
    thresh, smoothed = cv2.threshold(blur, 100, 255,cv2.THRESH_BINARY)
    polygon_mask = smoothed

    # transform to row col coordinates
    rowcol = np.flip(polygon_mask, axis=0)
    patch_origin = xy2rowcol(bounding_box[0], bounding_box[3], whole_canvas_height)
    
    retargeted[rowcol == 0] = 0
    return retargeted, patch_origin

def render_collage(input_image_collection_folder, output_dir, scaling_factor):

    with open(join(output_dir, 'slicing_result.json'), 'r') as f:
        layout = json.load(f)

    whole_canvas = np.zeros((layout['height']*scaling_factor, layout['width']*scaling_factor, 4), np.uint8)

    for img_dict in layout['images']:
        t_part = layout['parts'][img_dict['assigned_part']]

        patch, patch_origin = generate_image_patch(t_part, img_dict, input_image_collection_folder, whole_canvas.shape[0], magnification=scaling_factor)
        
        whole_canvas[patch_origin[0]:patch_origin[0]+patch.shape[0],patch_origin[1]:patch_origin[1]+patch.shape[1]] += patch
    #main_fig, main_ax = plt.subplots(nrows=1, ncols=1, num='Layout', figsize=(18, 18))
    write_color_image(whole_canvas, join(output_dir, 'collage.png'))

    # image with borders
    border = np.zeros((layout['height']*scaling_factor, layout['width']*scaling_factor), np.uint8)

    for cut in layout['cuts']:
        cv2.line(border, tuple(int(coord*scaling_factor) for coord in tuple(cut[0])), tuple(int(coord*scaling_factor) for coord in tuple(cut[1])), 255, 2*scaling_factor, cv2.LINE_AA, 0)
    border_flipped = np.flip(border, axis=0)
    height_border = border_flipped.shape[0]
    width_border = border_flipped.shape[1]

    canvas_border = whole_canvas.copy()

    canvas_border[border_flipped > 100] = np.array(np.broadcast_to(border_flipped.reshape(height_border, width_border, 1), (height_border, width_border, 4)))[border_flipped > 100]
    canvas_border[border_flipped > 100][:,3] = 255

    write_color_image(canvas_border, join(output_dir, 'collage_white_space.png'))

if __name__ == '__main__':
    input_image_collection_folder = sys.argv[1]
    output_dir = sys.argv[2]
    scaling_factor = int(sys.argv[3])
    render_collage(input_image_collection_folder, output_dir, scaling_factor)