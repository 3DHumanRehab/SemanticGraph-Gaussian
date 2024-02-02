# SEQUENCES=("my_377" "my_386" "my_387" "my_392" "my_393" "my_394")
SEQUENCES=("my_377")
# TYPES=("seg_label_clu_3d" "seg_label_clu", "seg_label", "seg")
TYPES=("clip_self_cloud_clu_depth_3d")
# for SEQUENCE in ${SEQUENCES[@]}; do
for TYPE in ${TYPES[@]}; do
    # dataset="../../zju_mocap_refine/$SEQUENCE"
    #dataset="../../zju_mocap_refine/my_377"
    dataset="/HOME/HOME/Zhangweiyue/gau2/data/zju_mocap_refine/my_377"
    # python train.py -s $dataset --eval --exp_name zju_mocap_refine/${SEQUENCE}_seg --motion_offset_flag --smpl_type smpl --actor_gender neutral --iterations 1200 --port 7000
     python train.py -s $dataset --eval --exp_name zju_mocap_refine/my_377_${TYPE} --motion_offset_flag --smpl_type smpl --actor_gender neutral --iterations 1200 --port 7005
done