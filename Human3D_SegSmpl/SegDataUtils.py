from plyfile import PlyData
import numpy as np
import os
import pickle

class SegHumanPlyData():
    def __init__(self) -> None:
        
        self.map_color2label=[
            [226, 226, 226],
            [158, 143, 20],  # rightHand
            [243, 115, 68],  # rightUpLeg
            [228, 162, 227], # leftArm
            [210, 78, 142],  # head
            [152, 78, 163],  # leftLeg
            [76 , 134, 26],  # leftFoot
            [100, 143, 255], # torso
            [129, 0  , 50],  # rightFoot
            [255, 176, 0],   # rightArm
            [192, 100, 119], # leftHand
            [149, 192, 228], # rightLeg
            [243, 232, 88],  # leftForeArm
            [90 , 64 , 210], # rightForeArm
            [152, 200, 156], # leftUpLeg
            [129, 103, 106], # hips
            ]
    
    def get_labled_ply(self,origin_ply_path, save_ply_path):
        ply = self.read_plyfile(origin_ply_path)
        points,colors=self.get_ply_points_and_colors(ply)

        points = self.fix_rotation(points)

        labels = [self.color2label(color) for color in colors]
        arr = [(points[i][0],points[i][1],points[i][2],
                colors[i][0],colors[i][1],colors[i][2],                
                labels[i]) for i in range(len(points))]

        arr = np.array(arr,dtype=[
            ('x','f4'),
            ('y','f4'),
            ('z','f4'),
            ('red','uint8'),
            ('green','uint8'),
            ('blue','uint8'),
            ('label','uint8')])

        from plyfile import PlyElement
        el = PlyElement.describe(arr,'vertex')
        plydata = PlyData([el],False,'<')

        self.save_plyfile(plydata,save_ply_path)

    def fix_rotation(self, points):
        
        ret = np.array( points)[:,[0,2,1]]
        ret[:,2] = -ret[:,2]
        return ret

    def color2label(self,_color:list):
        from operator import eq
        for i in range(len(self.map_color2label)):
            if eq(self.map_color2label[i],_color):
                return i
        return 0


    def read_plyfile(self,file_path):
        """Read ply file and return it as numpy array. Returns None if emtpy."""
        with open(file_path, "rb") as f:
            plydata = PlyData.read(f)
            return plydata
        
    def save_plyfile(self,plydata:PlyData,file_path):
        print(f'save to{file_path}')
        with open(file_path,"wb") as f:
            plydata.text=False
            plydata.write(f)
    
    def get_ply_points_and_colors(self,plydata:PlyData):
        length = plydata.elements[0].data.shape[0]
        ply_data = plydata.elements[0].data

        points=[]
        colors=[]
        for data in ply_data:
            points.append([data[0],data[1],data[2]])
            colors.append([data[6],data[7],data[8]])
        return points,colors
    
class SMPLModel():
  def __init__(self, model_path):
    """
    SMPL model.

    Parameter:
    ---------
    model_path: Path to the SMPL model parameters, pre-processed by
    `preprocess.py`.

    """
    with open(model_path, 'rb') as f:
      with open(model_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        params =  u.load()

      self.J_regressor = params['J_regressor']
      self.weights = params['weights']
      self.posedirs = params['posedirs']
      self.v_template = params['v_template']
      self.shapedirs = params['shapedirs']
      self.faces = params['f']
      self.kintree_table = params['kintree_table']

if __name__=='__main__':
    input_path = '/root/HumanGaussian_zwy/Human3D/saved/Human3D_test_set3/export/raw_test_set_smpl/predictions_mhbps.ply'
    output_path = '/root/HumanGaussian_zwy/SegmentHuman/smpl_predict.ply'
    seg = SegHumanPlyData()
    seg.get_labled_ply(input_path,output_path)