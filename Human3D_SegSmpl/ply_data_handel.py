from plyfile import PlyData
import numpy as np
import os

def read_plyfile(file_path):
    """Read ply file and return it as numpy array. Returns None if emtpy."""
    with open(file_path, "rb") as f:
        plydata = PlyData.read(f)
        return plydata

def save_plyfile(plydata:PlyData,file_path):
    print(f'save to{file_path}')
    with open(file_path,"wb") as f:
        plydata.text=False
        plydata.write(f)

def generate_plydata(data:list):

    arr = [(d[0],d[1],d[2],255,255,255,1,125) for d in data]

    arr = np.array(arr,dtype=[
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
    return plydata

def get_ply_points(gaussian_ply:PlyData):
    length = gaussian_ply.elements[0].data.shape[0]
    gau_data = gaussian_ply.elements[0].data
    
    points = [[gau_data[i][0],gau_data[i][1],gau_data[i][2]] for i in range(0,length)]
    return points

def gaussian_to_ply_change_axis(gaussian_ply:PlyData):
    length = gaussian_ply.elements[0].data.shape[0]
    gau_data = gaussian_ply.elements[0].data
    
    points = [[gau_data[i][0],-gau_data[i][2],gau_data[i][1]] for i in range(0,length)]
    return points

def gaussian2ply(input_file_path,output_file_path):
    ply = read_plyfile(input_path)
    arr = gaussian_to_ply_change_axis(ply)

    arr = np.array(arr)
    for i in range(3):
        arr[:,i] -= np.mean(arr[:,i])
        print(np.max(arr[:,i]))

    ply = generate_plydata(arr)
    save_plyfile(ply,output_path)

def seperate_ply(points,axis=1,offset=0.0):
    ret = [[],[]]
    for d in points:
        ri=0
        if d[axis]>=offset:
            ri=1
        ret[ri].append([d[0],d[1],d[2]])
    return {'front':ret[0],'back':ret[1]}

def can_add_point(new_point, exist_point):
    length = abs(new_point[1] - exist_point[1])
    radius=0.2
    if (new_point[0]-exist_point[0])**2 + (new_point[2]-exist_point[2])**2 > (radius*length)**2 :
        return True
    else:
        return False


def projection(points,axis=1):
    points = np.array(points)
    ordered_points = points[np.lexsort((points[:,2],points[:,0],points[:,1]))]
    project_points=[]
    i=0
    for p in ordered_points:
        f = True
        for pp in project_points:
            if can_add_point(p,pp)==False:
                f=False
                break
        if f==True:
            project_points.append(p)
        if i%500==0:
            print(f'{i}/{len(points)} project:{len(project_points)}')
        i+=1
    print(f'size of project:{len(project_points)}')
    return project_points

input_path = '/root/HumanGaussian_zwy/Human3D/point_cloud.ply'
output_path = '/root/HumanGaussian_zwy/Human3D/data/raw/smpl_gaussian_data/smpl_gaussian.ply'
dir_path = '/root/HumanGaussian_zwy/Human3D/data/raw/smpl_gaussian_data/'

def fix_coor_bug_for_sample():
    ply = read_plyfile('/root/HumanGaussian_zwy/Human3D/data/raw/egobody_dif/egobody_sample_single_human_no_background_no_color.ply')
    points = get_ply_points(ply)
    points = np.array(points)
    coords = points[:, [0, 2, 1]]
    coords[:, 2] = -coords[:, 2]
    ply = generate_plydata(coords)
    save_plyfile(ply,os.path.join(dir_path,'sample_human.ply'))

def check_seg_result(input_path,output_dir):
    ply = read_plyfile(input_path)
    print(ply)
    length = ply.elements[0].data.shape[0]
    gau_data = ply.elements[0].data

    label2part = ['background',
                  'rightHand' ,
                  'rightUpLeg',
                  'leftArm'   ,
                  'head'      ,
                  'leftLeg'   ,
                  'leftFoot'  ,
                  'torso'     ,
                  'rightFoot' ,
                  'rightArm'   ,
                  'leftHand'  ,
                  'rightLeg'  ,
                  'leftForeArm',
                  'rightForeArm',
                  'leftUpLeg',
                  'hips']
    part_data={}
    for part_name in label2part:
        part_data[part_name]=[]
    for line in gau_data:
        part_name = label2part[line[6]]
        part_data[part_name].append((line[0],line[1],line[2],line[3],line[4],line[5],line[6]))

    for key,value in part_data.items():
        print(f'{key}: {len(value)}')
    
    for part,data in part_data.items():
        save_path = os.path.join(output_dir,part+'.ply')
        arr = np.array(data,dtype=[
            ('x','f4'),
            ('y','f4'),
            ('z','f4'),
            ('red','uint8'),
            ('green','uint8'),
            ('blue','uint8'),
            ('part_label','uint8')])

        from plyfile import PlyElement
        el = PlyElement.describe(arr,'vertex')
        plydata = PlyData([el],False,'<')
        save_plyfile(plydata,save_path)
        

if __name__=='__main__':
    check_seg_result('/root/HumanGaussian_zwy/smpl_gaussian.ply','/root/HumanGaussian_zwy/smpl_gaussian')
