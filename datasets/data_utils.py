import numpy as np


def read_point_cloud_off(path):
    with open(path, 'r') as file:
        off_header = file.readline().strip()
        if 'OFF' == off_header:
            n_vertices, _, _ = tuple([int(s) for s in file.readline().strip().split(' ')])
        else:
            n_vertices, _, _ = tuple([int(s) for s in off_header[3:].split(' ')])
        vertices = [[float(s) for s in file.readline().strip().split(' ')] for _ in range(n_vertices)]
        # faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return np.array(vertices, np.float32)
