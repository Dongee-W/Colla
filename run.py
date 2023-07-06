import shape_decomposition as sd
import sas_optimization as so
import collage_assembly as ca
import sys

if __name__ == '__main__':
    input_shape = sys.argv[1]
    input_mask_folder = sys.argv[2]
    input_image_collection_folder = sys.argv[3]
    output_dir = sys.argv[4]
    scaling_factor = int(sys.argv[5])


    sd.generate_cuts(input_shape, output_dir)
    so.optimization(input_shape, input_mask_folder, output_dir)
    ca.render_collage(input_image_collection_folder, output_dir, scaling_factor)