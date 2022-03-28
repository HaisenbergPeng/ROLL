#-*- coding: utf-8 -*-  
# import pcl
import numpy as np

import matplotlib.pyplot as plt
import struct
import pypcd
import lzf

# def parse_binary_compressed_pc_data(f, dtype, metadata):
#     """ Parse lzf-compressed data.
#     Format is undocumented but seems to be:
#     - compressed size of data (uint32)
#     - uncompressed size of data (uint32)
#     - compressed data
#     - junk
#     """
#     fmt = 'II'
#     compressed_size, uncompressed_size =\
#         struct.unpack(fmt, f.read(struct.calcsize(fmt)))
#     compressed_data = f.read(compressed_size)
#     # TODO what to use as second argument? if buf is None
#     # (compressed > uncompressed)
#     # should we read buf as raw binary?
#     buf = lzf.decompress(compressed_data, uncompressed_size)
#     if len(buf) != uncompressed_size:
#         raise IOError('Error decompressing data')
#     # the data is stored field-by-field
#     pc_data = np.zeros(metadata['width'], dtype=dtype)
#     ix = 0
#     for dti in range(len(dtype)):
#         dt = dtype[dti] 
#         bytes = dt.itemsize * metadata['width']
#         column = np.fromstring(buf[ix:(ix+bytes)], dt)
#         pc_data[dtype.names[dti]] = column
#         ix += bytes
#     return pc_data

def main():
    # # only xyz,xyzi,xyzrgb is allowed
    # # cannot read timestamp
    filename = '/mnt/sdb/Datasets/DaoxiangLake/tests/pcd/7569.pcd'

    ########################################################################
    # using python-pcl
    ########################################################################
    # data = pcl.load(filename)  
    # print(data[0][0],data[0][1],data[0][2] )  

    ########################################################################
    # using pypcd:   assert(len(self.pc_data) == self.points)
    # using pypcd in py3.6: AttributeError: module 'pypcd' has no attribute 'PointCloud'
    ########################################################################
    # also can read from file handles.
    pc = pypcd.PointCloud.from_path(filename)
    # pc.pc_data has the data as a structured array
    # pc.fields, pc.count, etc have the metadata
    # center the x field
    # pc.pc_data['x'] -= pc.pc_data['x'].mean()
    print('fields: ', pc.fields)
    print('count: ', pc.count)

    # pts = data.to_array().transpose()
    # print(pts.shape)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    
    # x = pts[0]
    # y = pts[1]
    # z = pts[2]

    # ax.scatter(x, y, z, c='r', marker='o')
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')

    # # plt.show()


    # #######################################
    # # forget it: it is a compressed binary form using LZF compression
    # f = open(filename, 'rb')
    # header = f.read(216)
    # print(header)
    # # packet = f.read(21)
    # # xyzit = []
    # # print(packet)
    
    # x = struct.unpack('f',f.read(4))[0]
    # print(x)
    # f.close()
    # WIDTH: 64 point per line
    # for i in range(10):
    #     x,y,z,i,t = struct.unpack('3fBd',f.read(24))
    #     # f.read(1)
    #     # y = struct.unpack('f',f.read(4))[0]
    #     # f.read(1)
    #     # z = struct.unpack('f',f.read(4))[0]
    #     # f.read(1)
    #     # i = struct.unpack('B',f.read(1))[0]
    #     # f.read(1)
    #     # t = struct.unpack('Q',f.read(8))[0]
    #     # f.read(2)
    #     print(x,y,z,i,t)

if __name__ == "__main__":
    main()

# from pathlib import Path

# import numpy as np
# import open3d as o3d
# from traceback import format_exc


# def read_pcd(folder):
#     pcd = o3d.io.read_point_cloud(folder)
#     help(pcd)

# if __name__ == '__main__':
#     pcd_file_path = '/mnt/sdb/Datasets/DaoxiangLake/apollo_daoxianglake_demo/demo_data/pcd_data/20190918143332/50000.pcd'
#     read_pcd(pcd_file_path)
