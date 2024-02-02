# SEQUENCES=("my_377" "my_386" "my_387" "my_392" "my_393" "my_394")
SEQUENCES=("my_377")
TYPES=("seg_label_clu_3d" "seg_label_clu", "seg_label", "seg")
for TYPE in ${TYPES[@]}; do
    python render.py -m output/zju_mocap_refine/my_377_${TYPE} --motion_offset_flag --smpl_type smpl --actor_gender neutral --iteration 1200 --skip_train
done
/HOME/HOME/Zhangweiyue/gau2/output/zju_mocap_refine/my_377_clip_cloud_depth_3d_iter900