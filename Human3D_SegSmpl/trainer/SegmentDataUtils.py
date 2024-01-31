from plyfile import PlyData
import numpy as np

def read_plyfile(file_path):
    """Read ply file and return it."""
    with open(file_path, "rb") as f:
        plydata = PlyData.read(f)
        return plydata
    
def get_points_and_SegLabels(plydata:PlyData):
    length = plydata.elements[0].data.shape[0]
    gau_data = plydata.elements[0].data
    points = [[gau_data[i][0],gau_data[i][1],gau_data[i][2]] for i in range(0,length)]
    return points

def save_plyfile(plydata:PlyData,save_path):
    #print(f'save to{save_path}')
    with open(save_path,"wb") as f:
        plydata.text=False
        plydata.write(f)

def generate_plydata_with_Points_Colors_SegLabels(points:np.ndarray, colors:np.ndarray, seg_labels:np.ndarray):
    #assert points.shape[0]==seg_labels.shape[0]
    arr = [(points[i][0],points[i][1],points[i][2],
            colors[i][0],colors[i][1],colors[i][2],seg_labels[i]) for i in range(len(points))]
    arr = np.array(arr,dtype=[
        ('x','f4'),
        ('y','f4'),
        ('z','f4'),
        ('red','uint8'),
        ('green','uint8'),
        ('blue','uint8'),
        ('seg_label','uint8'),])
    from plyfile import PlyElement
    el = PlyElement.describe(arr,'vertex')
    plydata = PlyData([el],False,'<')
    return plydata