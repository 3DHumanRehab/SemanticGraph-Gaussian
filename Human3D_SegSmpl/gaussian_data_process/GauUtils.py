from plyfile import PlyData
import numpy as np
import os

def read_plyfile(file_path):
    """Read ply file and return it as numpy array. Returns None if emtpy."""
    with open(file_path, "rb") as f:
        plydata = PlyData.read(f)
        print('read in:')
        print(plydata)
        print(plydata.elements[0].data[0])
        print('\n')
        return plydata

def save_plyfile(plydata:PlyData,file_path):
    print(plydata)
    with open(file_path,"wb") as f:
        plydata.text=False
        plydata.write(f)

def generate_plydata(data:list):
    arr = np.array(data,dtype=[
        ('x','f4'),
        ('y','f4'),
        ('z','f4'),
        ('red','uint8'),
        ('green','uint8'),
        ('blue','uint8'),
        ('inst_label','uint8'),
        ('label','uint8'),])

    from plyfile import PlyElement
    el = PlyElement.describe(arr,'vertex')
    plydata = PlyData([el],False,'<')
    print(plydata)
    print(plydata.elements[0].data[0])
    return plydata

def gaussian_to_ply(gaussian_ply:PlyData, is_need_background:bool=False):
    length = gaussian_ply.elements[0].data.shape[0]
    gau_data = gaussian_ply.elements[0].data
    
    points = [(gau_data[i][0],gau_data[i][1],gau_data[i][2],
               0,0,0,
                1,125 ) for i in range(0,length)]
    return points

def get_human(human_ply:PlyData, is_seg_part:bool=False):
    points = []
    length = human_ply.elements[0].data.shape[0]
    human_data = human_ply.elements[0].data

    if is_seg_part==True:
        for d in human_data:
            if d[6]==1:
                points.append((d[0],d[1],d[2],0,0,0,1,d[7]))
    else:
        for d in human_data:
            if d[6]==1:
                points.append((d[0],d[1],d[2],0,0,0,1,125))
    return points

def get_human_with_background(human_ply:PlyData):
    points = []
    length = human_ply.elements[0].data.shape[0]
    human_data = human_ply.elements[0].data

    for d in human_data:
        if d[6]==1:
            points.append((d[0],d[1],d[2],0,0,0,1,d[7]))
        elif d[6]==0:
            points.append((d[0],d[1],d[2],0,0,0,0,d[7]))
    return points

def get_human_with_color(human_ply:PlyData, is_seg_part:bool=False):
    points = []
    length = human_ply.elements[0].data.shape[0]
    human_data = human_ply.elements[0].data

    if is_seg_part==True:
        for d in human_data:
            if d[6]==1:
                points.append((d[0],d[1],d[2],d[3],d[4],d[5],1,d[7]))
    else:
        for d in human_data:
            if d[6]==1:
                points.append((d[0],d[1],d[2],d[3],d[4],d[5],1,125))
    return points

def get_no_color(human_ply:PlyData):
    points = [(d[0],d[1],d[2],0,0,0,d[6],d[7]) for d in human_ply.elements[0].data]
    return points


if __name__=='__main__':
    print(__file__)