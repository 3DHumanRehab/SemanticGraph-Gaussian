SEQUENCES=("my_377" "my_386" "my_387" "my_392" "my_393" "my_394")
for SEQUENCE in ${SEQUENCES[@]}; do
    dataset="data/zju_mocap_refine/$SEQUENCE"
    python train.py -s $dataset --eval --exp_name zju_mocap_refine/${SEQUENCE} --motion_offset_flag --smpl_type smpl --actor_gender neutral --iterations 1200 
done