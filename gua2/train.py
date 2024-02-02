#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os

from scene.dataset_readers import fetchPly
from utils.graph_utils import node_clustering, point_to_graph, nodeEmbedding_node2vec
from utils.pcd import HumanSegmentationDataset

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
from random import randint
from utils.loss_utils import *
from utils.tensor_utils import normalize_tensor,change_tensor
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.graphics_utils import world_to_camera, world_to_pixel
from utils.image_utils import *
import uuid
import imageio
import torchvision
import numpy as np
import cv2
import pickle
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import lpips
loss_fn_vgg = lpips.LPIPS(net='vgg').to(torch.device('cuda', torch.cuda.current_device()))
import time
import torch.nn.functional as F

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, loss_type,coarse_iteration, subgraph_sample, graph_sample):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, dataset.smpl_type, dataset.motion_offset_flag, dataset.actor_gender)
    scene = Scene(dataset, gaussians)
    cls_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    Ll1_loss_for_log = 0.0
    mask_loss_for_log = 0.0
    ssim_loss_for_log = 0.0
    lpips_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # lpips_test_lst = []
    elapsed_time = 0
    pcd_path = 'smpl_predict.ply'  # sample input scene
    file_list = [pcd_path]  # for now just the demo scene
    pre_dataset = HumanSegmentationDataset(file_list=file_list)
    coords, colors, labels = pre_dataset.load_pc(pcd_path)
    gaussians.frozen_labels = labels.cuda()
    gaussians._objects_dc = F.one_hot(gaussians.frozen_labels.to(torch.int64),num_classes=20).unsqueeze(1).to(torch.float32)
    # frozen_labels = torch.argmax(gaussians._objects_dc.squeeze(1), dim=1)
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        # Start timer
        start_time = time.time()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        # gaussians._objects_dc = F.one_hot(gaussians.frozen_labels.to(torch.int64), num_classes=20).unsqueeze(1).to(torch.float32)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, alpha, viewspace_point_tensor, visibility_filter, radii,objects = render_pkg["render"], render_pkg["render_alpha"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["render_object"]
        # image[3,512,512], alpha[1,512,512], radii=[6890], objects[20,512,512]
        # Loss
        gt_image = viewpoint_cam.original_image.cuda() # [3,512,512]
        bkgd_mask = viewpoint_cam.bkgd_mask.cuda() # [1, 512, 512]
        bound_mask = viewpoint_cam.bound_mask.cuda() # [1, 512, 512]
        Ll1 = l1_loss(image.permute(1,2,0)[bound_mask[0]==1], gt_image.permute(1,2,0)[bound_mask[0]==1])
        mask_loss = l2_loss(alpha[bound_mask==1], bkgd_mask[bound_mask==1])
        # crop the object region
        x, y, w, h = cv2.boundingRect(bound_mask[0].cpu().numpy().astype(np.uint8))
        img_pred = image[:, y:y + h, x:x + w].unsqueeze(0)
        img_gt = gt_image[:, y:y + h, x:x + w].unsqueeze(0)
        # ssim loss
        ssim_loss = ssim(img_pred, img_gt)
        # lipis loss
        lpips_loss = loss_fn_vgg(img_pred, img_gt).reshape(-1)
        loss = Ll1 + 0.1 * mask_loss + 0.01 * (1.0 - ssim_loss) + 0.01 * lpips_loss
        if iteration%10==0:
            print("gua_loss={}".format(loss))
            print(f'\niteration: {iteration}/{opt.iterations}\n')

        # Semantic supervision loss
        frozen_labels = torch.argmax(gaussians._objects_dc.squeeze(1), dim=1)
        if iteration > coarse_iteration:
            if "3d" in loss_type and iteration % 100 == 0:
                obj_3d_loss = loss_cls_3d(gaussians._xyz.squeeze().detach(), gaussians._objects_dc.squeeze(1).detach())
                print("obj_3d_loss={}".format(obj_3d_loss))
                loss += obj_3d_loss
            if "cloud" in loss_type:
                cloud_loss = pointcloud_loss(torch.tensor(fetchPly(os.path.join("output", loss_type, "input.ply")).points).cuda(),gaussians._xyz)
                print("cloud_loss={}".format(cloud_loss))
                loss += cloud_loss * 1.0 * np.exp(-0.1 * iteration)
            if "clu" in loss_type:
                distance_tensor, scale_tensor = None, radii
                for i in range(gaussians._xyz.shape[0]):
                    camera_pos = world_to_camera(gaussians._xyz[i], torch.from_numpy(viewpoint_cam.R).float().cuda(),torch.from_numpy(viewpoint_cam.T).float().cuda())
                    distance = torch.norm(torch.from_numpy(viewpoint_cam.T).float().cuda() - camera_pos)
                    distance_tensor = distance.unsqueeze(0) if distance_tensor is None else torch.cat((distance_tensor, distance.unsqueeze(0)), dim=0)
                scale_tensor, distance_tensor = normalize_tensor(scale_tensor.float()), normalize_tensor(distance_tensor)
                weight_tensor = scale_tensor + distance_tensor
                reshaped_tensor = gaussians._features_rest.view(15, gaussians._xyz.shape[0], 3)
                matrix_list = [reshaped_tensor[i, :, :] for i in range(15)]
                features = torch.cat((gaussians._xyz, gaussians._features_dc.squeeze(1), gaussians._opacity, gaussians._rotation), dim=1)
                for x in matrix_list:
                    features = torch.cat((features, x), dim=1)
                index = torch.randperm(gaussians._xyz.shape[0] - 1)[0:graph_sample]
                graph = point_to_graph(gaussians._xyz[index].detach().cpu().numpy(),features[index].detach().cpu().numpy(), 3)
                data = nodeEmbedding_node2vec(graph).cuda()
                clu_loss = None
                for i in range(0, data.shape[0]):
                    for j in range(i + 1, data.shape[0]):
                        if frozen_labels[index][i] == frozen_labels[index][j]:
                            clu_loss = torch.nn.functional.cosine_similarity(data[i], data[j], dim=0) * weight_tensor[index][i] * weight_tensor[index][j] if clu_loss is None else clu_loss + torch.nn.functional.cosine_similarity(data[i], data[j], dim=0) * weight_tensor[index][i] * weight_tensor[index][j]
                        else:
                            clu_loss = - torch.nn.functional.cosine_similarity(data[i], data[j], dim=0) * weight_tensor[index][i] * weight_tensor[index][j] if clu_loss is None else clu_loss - torch.nn.functional.cosine_similarity(data[i], data[j], dim=0) * weight_tensor[index][i] * weight_tensor[index][j]
                print("clu_loss={}".format(clu_loss / graph_sample))
                loss += clu_loss / graph_sample
            if "depth" in loss_type:
                depth_loss = loss_depth(get_depth(gt_image.permute(1, 2, 0).cpu().detach().numpy()),get_depth(image.permute(1, 2, 0).cpu().detach().numpy()))
                print("depth_loss={}".format(depth_loss))
                loss += depth_loss
            if "self" in loss_type:
                point_dist_dic = {}
                point_attr_dic = {}
                point_color_dic = {}
                point_opacity_dic = {}
                for i in range(frozen_labels.shape[0]):
                    label = frozen_labels[i].item()
                    if point_attr_dic.get(label) is None:
                        point_dist_dic[label], point_color_dic[label], point_opacity_dic[label] = gaussians._xyz[i].unsqueeze(0), gaussians._features_dc[i], gaussians._opacity[i].unsqueeze(0)
                        point_attr_dic[label] = torch.cat((gaussians._features_dc[i].squeeze(1), gaussians._opacity[i].unsqueeze(0)), dim=1)
                    else:
                        point_dist_dic[label] = torch.cat((point_dist_dic[label], gaussians._xyz[i].unsqueeze(0)),dim=0)
                        point_color_dic[label] = torch.cat((point_color_dic[label], gaussians._features_dc[i]), dim=0)
                        point_opacity_dic[label] = torch.cat((point_opacity_dic[label], gaussians._opacity[i].unsqueeze(0)), dim=0)
                        point_attr_dic[label] = torch.cat((point_attr_dic[label], torch.cat((gaussians._features_dc[i].squeeze(1), gaussians._opacity[i].unsqueeze(0)), dim=1)), dim=0)
                self_loss = None
                for key in point_attr_dic.keys():
                    if point_attr_dic[key].shape[0] <= subgraph_sample:
                        detail_label = torch.tensor(node_clustering(point_dist_dic[key].detach().cpu().numpy(),point_attr_dic[key].detach().cpu().numpy())).view(1,-1).transpose(1, 0)
                        self_loss = loss_self_supervision(detail_label, point_color_dic[key], point_opacity_dic[key],point_dist_dic[key]) if self_loss is None else self_loss + loss_self_supervision(detail_label, point_color_dic[key], point_opacity_dic[key], point_dist_dic[key])
                    else:
                        index = torch.randperm(point_dist_dic[key].shape[0]-1)[0:subgraph_sample]
                        detail_label = torch.tensor(node_clustering(point_dist_dic[key][index].detach().cpu().numpy(),point_attr_dic[key][index].detach().cpu().numpy())).view(1, -1).transpose(1, 0)
                        self_loss = loss_self_supervision(detail_label, point_color_dic[key][index], point_opacity_dic[key][index],point_dist_dic[key][index]) if self_loss is None else self_loss + loss_self_supervision(detail_label, point_color_dic[key][index], point_opacity_dic[key][index], point_dist_dic[key][index])
                print("self_loss={}".format(self_loss))
                loss += self_loss
            if "clip" in loss_type:
                types_mapping = {0: "background",1: "rightHand",2: "rightUpLeg",3: "leftArm",4: "head",5: "leftLeg",6: "leftFoot",7: "torso",8: "rightFoot",9: "rightArm",10: "leftHand",11: "rightLeg",12: "leftForeArm",13: "rightForeArm",14: "leftUpLeg",15: "hips"}
                clip_loss = None
                object_type = torch.argmax(objects.permute(1, 2, 0), dim=2)
                for i in types_mapping.keys():
                    object_mask = (object_type == i)
                    img_clip_loss = img2img_clip_loss((image * object_mask).unsqueeze(0),(gt_image * object_mask).unsqueeze(0))
                    text_clip_loss = img2text_clip_loss("a image of " + types_mapping[i],(image * object_mask).unsqueeze(0))
                    clip_loss = img_clip_loss + text_clip_loss if clip_loss is None else clip_loss + img_clip_loss + text_clip_loss
                print("clip_loss={}".format(clip_loss))
                loss += clip_loss
        loss.backward()
        # print("my_loss={}".format(loss))

        # with open('label.txt', 'a') as file:
        #     file.write('{} and {} iteration with loss = {}\n'.format(loss_type,iteration,torch.mean(torch.mean(cls_criterion(seg_objects, gt_seg.long()))).item()))


        # end time
        end_time = time.time()
        # Calculate elapsed time
        elapsed_time += (end_time - start_time)

        if (iteration in testing_iterations):
            print("[Elapsed time]: ", elapsed_time)

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            Ll1_loss_for_log = 0.4 * Ll1.item() + 0.6 * Ll1_loss_for_log
            mask_loss_for_log = 0.4 * mask_loss.item() + 0.6 * mask_loss_for_log
            ssim_loss_for_log = 0.4 * ssim_loss.item() + 0.6 * ssim_loss_for_log
            lpips_loss_for_log = 0.4 * lpips_loss.item() + 0.6 * lpips_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"#pts": gaussians._xyz.shape[0], "Ll1 Loss": f"{Ll1_loss_for_log:.{3}f}", "mask Loss": f"{mask_loss_for_log:.{2}f}",
                                          "ssim": f"{ssim_loss_for_log:.{2}f}", "lpips": f"{lpips_loss_for_log:.{2}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Start timer
            start_time = time.time()
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, kl_threshold=0.4, t_vertices=viewpoint_cam.big_pose_world_vertex, iter=iteration)
                    # gaussians.densify_and_prune(opt.densify_grad_threshold, 0.01, scene.cameras_extent, 1)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # end time
            end_time = time.time()
            # Calculate elapsed time
            elapsed_time += (end_time - start_time)

            # if (iteration in checkpoint_iterations):
            if (iteration in testing_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):
    if not args.model_path:
        # if os.getenv('OAR_JOB_ID'):
        #     unique_str=os.getenv('OAR_JOB_ID')
        # else:
        #     unique_str = str(uuid.uuid4())
        # args.model_path = os.path.join("./output/", unique_str[0:10])
        args.model_path = os.path.join("./output/", args.exp_name)


    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        # validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
        #                       {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})

        smpl_rot = {}
        smpl_rot['train'], smpl_rot['test'] = {}, {}
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    smpl_rot[config['name']][viewpoint.pose_id] = {}
                    render_output = renderFunc(viewpoint, scene.gaussians, *renderArgs, return_smpl_rot=True)
                    image = torch.clamp(render_output["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    bound_mask = viewpoint.bound_mask
                    image.permute(1,2,0)[bound_mask[0]==0] = 0 if renderArgs[1].sum().item() == 0 else 1
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips_test += loss_fn_vgg(image, gt_image).mean().double()

                    smpl_rot[config['name']][viewpoint.pose_id]['transforms'] = render_output['transforms']
                    smpl_rot[config['name']][viewpoint.pose_id]['translation'] = render_output['translation']

                l1_test /= len(config['cameras'])
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {} #{}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], len(config['cameras']), l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        # Store data (serialize)
        save_path = os.path.join(scene.model_path, 'smpl_rot', f'iteration_{iteration}')
        os.makedirs(save_path, exist_ok=True)
        with open(save_path+"/smpl_rot.pickle", 'wb') as handle:
            pickle.dump(smpl_rot, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=7004)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[500,700,900,1200, 2000, 3000, 7000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[500,700,900,1200, 2000, 3000, 7000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[500,700,900,1200])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--coarse_iteration", type=int, default=500)
    parser.add_argument("--subgraph_sample", type=int, default=30)
    parser.add_argument("--graph_sample", type=int, default=50)
    
    args = parser.parse_args(sys.argv[1:])
    args.test_iterations.append(args.iterations)
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from,args.exp_name,args.coarse_iteration,args.subgraph_sample,args.graph_sample)
    # All done
    print("\nTraining complete.")
