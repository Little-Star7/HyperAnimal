python train.py --name tiger_resnet50 --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/stylegan3-data/tiger-train
python train.py --name tiger_convnext --use_convnext --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/stylegan3-data/tiger-train
python train.py --name tiger_swinv2 --use_swinv2 --lr 0.03 --warm_epoch 5 --erasing_p 0.5 --circle --data_dir /home/moon/yy/reid-animal/stylegan3-data/tiger-train
python train.py --name tiger_efficient --use_efficient --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/stylegan3-data/tiger-train
python train.py --name tiger_hr --use_hr --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.01 --circle --data_dir /home/moon/yy/reid-animal/stylegan3-data/tiger-train
python train.py --name tiger_dense --use_dense --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/stylegan3-data/tiger-train


#python train.py --name giant_resnet50 --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/stylegan3-data/giant-train
#python train.py --name giant_convnext --use_convnext --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/stylegan3-data/giant-train
#python train.py --name giant_swinv2 --use_swinv2 --lr 0.03 --warm_epoch 5 --erasing_p 0.5 --circle --data_dir /home/moon/yy/reid-animal/stylegan3-data/giant-train
#python train.py --name giant_efficient --use_efficient --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/stylegan3-data/giant-train
#python train.py --name giant_hr --use_hr --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.01 --circle --data_dir /home/moon/yy/reid-animal/stylegan3-data/giant-train
#python train.py --name giant_dense --use_dense --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/stylegan3-data/giant-train

#python train.py --name red_resnet50 --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/stylegan3-data/red-train
#python train.py --name red_convnext --use_convnext --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/stylegan3-data/red-train
#python train.py --name red_swinv2 --use_swinv2 --lr 0.03 --warm_epoch 5 --erasing_p 0.5 --circle --data_dir /home/moon/yy/reid-animal/stylegan3-data/red-train
#python train.py --name red_efficient --use_efficient --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/stylegan3-data/red-train
#python train.py --name red_hr --use_hr --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.01 --circle --data_dir /home/moon/yy/reid-animal/stylegan3-data/red-train
#python train.py --name red_dense --use_dense --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/stylegan3-data/red-train


#python train.py --name atrw_nocondition_resnet50 --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/atrw_nocondition
#python train.py --name atrw_nocondition_convnext --use_convnext --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/atrw_nocondition
#python train.py --name atrw_nocondition_swinv2 --use_swinv2 --lr 0.03 --warm_epoch 5 --erasing_p 0.5 --circle --data_dir /home/moon/yy/reid-animal/database/atrw_nocondition
#python train.py --name atrw_nocondition_efficient --use_efficient --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/atrw_nocondition
#python train.py --name atrw_nocondition_hr --use_hr --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.01 --circle --data_dir /home/moon/yy/reid-animal/database/atrw_nocondition
#python train.py --name atrw_nocondition_dense --use_dense --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/atrw_nocondition

#python train.py --name atrw_dropout_resnet50 --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/atrw_dropout
#python train.py --name atrw_dropout_convnext --use_convnext --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/atrw_dropout
#python train.py --name atrw_dropout_swinv2 --use_swinv2 --lr 0.03 --warm_epoch 5 --erasing_p 0.5 --circle --data_dir /home/moon/yy/reid-animal/database/atrw_dropout
#python train.py --name atrw_dropout_efficient --use_efficient --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/atrw_dropout
#python train.py --name atrw_dropout_hr --use_hr --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.01 --circle --data_dir /home/moon/yy/reid-animal/database/atrw_dropout
#python train.py --name atrw_dropout_dense --use_dense --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/atrw_dropout
#
#python train.py --name atrw_syn_resnet50 --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/atrw_syn
#python train.py --name atrw_syn_convnext --use_convnext --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/atrw_syn
#python train.py --name atrw_syn_swinv2 --use_swinv2 --lr 0.03 --warm_epoch 5 --erasing_p 0.5 --circle --data_dir /home/moon/yy/reid-animal/database/atrw_syn
#python train.py --name atrw_syn_efficient --use_efficient --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/atrw_syn
#python train.py --name atrw_syn_hr --use_hr --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.01 --circle --data_dir /home/moon/yy/reid-animal/database/atrw_syn
#python train.py --name atrw_syn_dense --use_dense --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/atrw_syn

#python train.py --name pre_gp_resnet50 --pre_train --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/iPanda-train
#python train.py --name pre_gp_convnext --pre_train --use_convnext --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/iPanda-train
#python train.py --name pre_gp_swinv2 --pre_train --use_swinv2 --lr 0.03 --warm_epoch 5 --erasing_p 0.5 --circle --data_dir /home/moon/yy/reid-animal/database/iPanda-train
#python train.py --name pre_gp_efficient --pre_train --use_efficient --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/iPanda-train
#python train.py --name pre_gp_hr --pre_train --use_hr --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.01 --circle --data_dir /home/moon/yy/reid-animal/database/iPanda-train
#python train.py --name pre_gp_dense --pre_train --use_dense --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/iPanda-train

#python train.py --name pre_atrw_resnet50 --pre_train --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/atrw-train
#python train.py --name pre_atrw_convnext --pre_train --use_convnext --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/atrw-train
#python train.py --name pre_atrw_swinv2 --pre_train --use_swinv2 --lr 0.03 --warm_epoch 5 --erasing_p 0.5 --circle --data_dir /home/moon/yy/reid-animal/database/atrw-train
#python train.py --name pre_atrw_efficient --pre_train --use_efficient --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/atrw-train
#python train.py --name pre_atrw_hr --pre_train --use_hr --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.01 --circle --data_dir /home/moon/yy/reid-animal/database/atrw-train
#python train.py --name pre_atrw_dense --pre_train --use_dense --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/atrw-train

#python train.py --name atrw_real_resnet50 --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/atrw-train
#python train.py --name atrw_real_convnext --use_convnext --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/atrw-train
#python train.py --name atrw_real_swinv2 --use_swinv2 --lr 0.03 --warm_epoch 5 --erasing_p 0.5 --circle --data_dir /home/moon/yy/reid-animal/database/atrw-train
#python train.py --name atrw_real_efficient --use_efficient --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/atrw-train
#python train.py --name atrw_real_hr --use_hr --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.01 --circle --data_dir /home/moon/yy/reid-animal/database/atrw-train
#python train.py --name atrw_real_dense --use_dense --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/atrw-train
#
#python train.py --name atrw_f1_resnet50 --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/atrw_f1
#python train.py --name atrw_f1_convnext --use_convnext --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/atrw_f1
#python train.py --name atrw_f1_swinv2 --use_swinv2 --lr 0.03 --warm_epoch 5 --erasing_p 0.5 --circle --data_dir /home/moon/yy/reid-animal/database/atrw_f1
#python train.py --name atrw_f1_efficient --use_efficient --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/atrw_f1
#python train.py --name atrw_f1_hr --use_hr --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.01 --circle --data_dir /home/moon/yy/reid-animal/database/atrw_f1
#python train.py --name atrw_f1_dense --use_dense --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/atrw_f1

#python train.py --name atrw_pre_real_resnet50 --pre_train --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/atrw-train

#python train.py --name atrw_f1_resnet50 --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/atrw_f1

#python train.py --name atrw_real_resnet50 --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/atrw-train
#python train.py --name ipanda_gp_f1_new_resnet50 --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/gp_f1_new

#python train.py --name pre_real_resnet50 --pre_train --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/iPanda-train

#python train.py --name ipanda_gp_f1_new_resnet50 --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/gp_f1
#python train.py --name ipanda_gp_f1_convnext --use_convnext --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/gp_f1
#python train.py --name ipanda_gp_f1_swinv2 --use_swinv2 --lr 0.03 --warm_epoch 5 --erasing_p 0.5 --circle --data_dir /home/moon/yy/reid-animal/database/gp_f1
#python train.py --name ipanda_gp_f1_efficient --use_efficient --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/gp_f1
#python train.py --name ipanda_gp_f1_hr --use_hr --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.01 --circle --data_dir /home/moon/yy/reid-animal/database/gp_f1
#python train.py --name ipanda_gp_f1_dense --use_dense --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/gp_f1
#
#python train.py --name ipanda_gp_syn_resnet50 --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/gp_syn
#python train.py --name ipanda_gp_syn_convnext --use_convnext --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/gp_syn
#python train.py --name ipanda_gp_syn_swinv2 --use_swinv2 --lr 0.03 --warm_epoch 5 --erasing_p 0.5 --circle --data_dir /home/moon/yy/reid-animal/database/gp_syn
#python train.py --name ipanda_gp_syn_efficient --use_efficient --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/gp_syn
#python train.py --name ipanda_gp_syn_hr --use_hr --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.01 --circle --data_dir /home/moon/yy/reid-animal/database/gp_syn
#python train.py --name ipanda_gp_syn_dense --use_dense --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/gp_syn
#
#python train.py --name ipanda_gp_dropout_resnet50 --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/gp_dropout
#python train.py --name ipanda_gp_dropout_convnext --use_convnext --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/gp_dropout
#python train.py --name ipanda_gp_dropout_swinv2 --use_swinv2 --lr 0.03 --warm_epoch 5 --erasing_p 0.5 --circle --data_dir /home/moon/yy/reid-animal/database/gp_dropout
#python train.py --name ipanda_gp_dropout_efficient --use_efficient --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/gp_dropout
#python train.py --name ipanda_gp_dropout_hr --use_hr --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.01 --circle --data_dir /home/moon/yy/reid-animal/database/gp_dropout
#python train.py --name ipanda_gp_dropout_dense --use_dense --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/gp_dropout



#python train.py --name ipanda_resnet50 --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/meta-iPanda-50/iPanda-all

#python train.py --name ipanda_real_resnet50 --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/iPanda-train
#python train.py --name ipanda_real_convnext --use_convnext --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/iPanda-train
#python train.py --name ipanda_real_swinv2 --use_swinv2 --lr 0.03 --warm_epoch 5 --erasing_p 0.5 --circle --data_dir /home/moon/yy/reid-animal/database/iPanda-train
#python train.py --name ipanda_real_efficient --use_efficient --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/iPanda-train
#python train.py --name ipanda_real_hr --use_hr --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.01 --circle --data_dir /home/moon/yy/reid-animal/database/iPanda-train
#python train.py --name ipanda_real_dense --use_dense --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/iPanda-train

#python train.py --name pre_real_resnet50 --pre_train --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/RedPanda-33
#python train.py --name pre_real_convnext --pre_train --use_convnext --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/RedPanda-33
#python train.py --name pre_real_swinv2 --pre_train --use_swinv2 --lr 0.03 --warm_epoch 5 --erasing_p 0.5 --circle --data_dir /home/moon/yy/reid-animal/database/RedPanda-33
#python train.py --name pre_real_efficient --pre_train --use_efficient --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/RedPanda-33
#python train.py --name pre_real_hr --pre_train --use_hr --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.01 --circle --data_dir /home/moon/yy/reid-animal/database/RedPanda-33
#python train.py --name pre_real_dense --pre_train --use_dense --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/RedPanda-33

#python train.py --name rp_f1_500_n20_resnet50 --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_500_n20
#python train.py --name rp_f1_500_n20_convnext --use_convnext --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_500_n20
#python train.py --name rp_f1_500_n20_swinv2 --use_swinv2 --lr 0.03 --warm_epoch 5 --erasing_p 0.5 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_500_n20
#python train.py --name rp_f1_500_n20_efficient --use_efficient --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_500_n20
#python train.py --name rp_f1_500_n20_hr --use_hr --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.01 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_500_n20
#python train.py --name rp_f1_500_n20_dense --use_dense --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_500_n20
#
#python train.py --name rp_f1_1000_n20_resnet50 --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_1000_n20
#python train.py --name rp_f1_1000_n20_convnext --use_convnext --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_1000_n20
#python train.py --name rp_f1_1000_n20_swinv2 --use_swinv2 --lr 0.03 --warm_epoch 5 --erasing_p 0.5 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_1000_n20
#python train.py --name rp_f1_1000_n20_efficient --use_efficient --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_1000_n20
#python train.py --name rp_f1_1000_n20_hr --use_hr --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.01 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_1000_n20
#python train.py --name rp_f1_1000_n20_dense --use_dense --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_1000_n20

#python train.py --name rp_f1_resnet50 --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1
#python train.py --name rp_f1_convnext --use_convnext --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1
#python train.py --name rp_f1_swinv2 --use_swinv2 --lr 0.03 --warm_epoch 5 --erasing_p 0.5 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1
#python train.py --name rp_f1_efficient --use_efficient --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1
#python train.py --name rp_f1_hr --use_hr --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.01 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1
#python train.py --name rp_f1_dense --use_dense --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1
#
#python train.py --name rp_f1_clip_resnet50 --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_clip
#python train.py --name rp_f1_clip_convnext --use_convnext --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_clip
#python train.py --name rp_f1_clip_swinv2 --use_swinv2 --lr 0.03 --warm_epoch 5 --erasing_p 0.5 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_clip
#python train.py --name rp_f1_clip_efficient --use_efficient --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_clip
#python train.py --name rp_f1_clip_hr --use_hr --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.01 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_clip
#python train.py --name rp_f1_clip_dense --use_dense --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_clip
#
#python train.py --name rp_f10_resnet50 --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f10
#python train.py --name rp_f10_convnext --use_convnext --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f10
#python train.py --name rp_f10_swinv2 --use_swinv2 --lr 0.03 --warm_epoch 5 --erasing_p 0.5 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f10
#python train.py --name rp_f10_efficient --use_efficient --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f10
#python train.py --name rp_f10_hr --use_hr --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.01 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f10
#python train.py --name rp_f10_dense --use_dense --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f10
#
#python train.py --name rp_f0p1_resnet50 --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f0p1
#python train.py --name rp_f0p1_convnext --use_convnext --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f0p1
#python train.py --name rp_f0p1_swinv2 --use_swinv2 --lr 0.03 --warm_epoch 5 --erasing_p 0.5 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f0p1
#python train.py --name rp_f0p1_efficient --use_efficient --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f0p1
#python train.py --name rp_f0p1_hr --use_hr --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.01 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f0p1
#python train.py --name rp_f0p1_dense --use_dense --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f0p1
#
#python train.py --name rp_f1_100_resnet50 --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_100
#python train.py --name rp_f1_100_convnext --use_convnext --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_100
#python train.py --name rp_f1_100_swinv2 --use_swinv2 --lr 0.03 --warm_epoch 5 --erasing_p 0.5 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_100
#python train.py --name rp_f1_100_efficient --use_efficient --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_100
#python train.py --name rp_f1_100_hr --use_hr --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.01 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_100
#python train.py --name rp_f1_100_dense --use_dense --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_100
#
#python train.py --name rp_f1_500_resnet50 --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_500
#python train.py --name rp_f1_500_convnext --use_convnext --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_500
#python train.py --name rp_f1_500_swinv2 --use_swinv2 --lr 0.03 --warm_epoch 5 --erasing_p 0.5 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_500
#python train.py --name rp_f1_500_efficient --use_efficient --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_500
#python train.py --name rp_f1_500_hr --use_hr --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.01 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_500
#python train.py --name rp_f1_500_dense --use_dense --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_500
#
#python train.py --name rp_f1_1000_resnet50 --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_1000
#python train.py --name rp_f1_1000_convnext --use_convnext --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_1000
#python train.py --name rp_f1_1000_swinv2 --use_swinv2 --lr 0.03 --warm_epoch 5 --erasing_p 0.5 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_1000
#python train.py --name rp_f1_1000_efficient --use_efficient --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_1000
#python train.py --name rp_f1_1000_hr --use_hr --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.01 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_1000
#python train.py --name rp_f1_1000_dense --use_dense --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_1000
#
#python train.py --name rp_f1_4000_resnet50 --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_4000
#python train.py --name rp_f1_4000_convnext --use_convnext --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_4000
#python train.py --name rp_f1_4000_swinv2 --use_swinv2 --lr 0.03 --warm_epoch 5 --erasing_p 0.5 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_4000
#python train.py --name rp_f1_4000_efficient --use_efficient --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_4000
#python train.py --name rp_f1_4000_hr --use_hr --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.01 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_4000
#python train.py --name rp_f1_4000_dense --use_dense --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_4000
#
#python train.py --name rp_f1_n20_resnet50 --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_n20
#python train.py --name rp_f1_n20_convnext --use_convnext --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_n20
#python train.py --name rp_f1_n20_swinv2 --use_swinv2 --lr 0.03 --warm_epoch 5 --erasing_p 0.5 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_n20
#python train.py --name rp_f1_n20_efficient --use_efficient --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_n20
#python train.py --name rp_f1_n20_hr --use_hr --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.01 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_n20
#python train.py --name rp_f1_n20_dense --use_dense --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_f1_n20
#
#python train.py --name real_resnet50 --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/RedPanda-33
#python train.py --name real_convnext --use_convnext --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/RedPanda-33
#python train.py --name real_swinv2 --use_swinv2 --lr 0.03 --warm_epoch 5 --erasing_p 0.5 --circle --data_dir /home/moon/yy/reid-animal/database/RedPanda-33
#python train.py --name real_efficient --use_efficient --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/RedPanda-33
#python train.py --name real_hr --use_hr --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.01 --circle --data_dir /home/moon/yy/reid-animal/database/RedPanda-33
#python train.py --name real_dense --use_dense --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/RedPanda-33
#
#python train.py --name rp_syn_resnet50 --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_syn
#python train.py --name rp_syn_convnext --use_convnext --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_syn
#python train.py --name rp_syn_swinv2 --use_swinv2 --lr 0.03 --warm_epoch 5 --erasing_p 0.5 --circle --data_dir /home/moon/yy/reid-animal/database/rp_syn
#python train.py --name rp_syn_efficient --use_efficient --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_syn
#python train.py --name rp_syn_hr --use_hr --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.01 --circle --data_dir /home/moon/yy/reid-animal/database/rp_syn
#python train.py --name rp_syn_dense --use_dense --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_syn
#
#python train.py --name rp_dropout_resnet50 --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_dropout
#python train.py --name rp_dropout_convnext --use_convnext --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_dropout
#python train.py --name rp_dropout_swinv2 --use_swinv2 --lr 0.03 --warm_epoch 5 --erasing_p 0.5 --circle --data_dir /home/moon/yy/reid-animal/database/rp_dropout
#python train.py --name rp_dropout_efficient --use_efficient --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_dropout
#python train.py --name rp_dropout_hr --use_hr --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.01 --circle --data_dir /home/moon/yy/reid-animal/database/rp_dropout
#python train.py --name rp_dropout_dense --use_dense --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_dropout
#
#python train.py --name rp_kl_resnet50 --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_kl
#python train.py --name rp_kl_convnext --use_convnext --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_kl
#python train.py --name rp_kl_swinv2 --use_swinv2 --lr 0.03 --warm_epoch 5 --erasing_p 0.5 --circle --data_dir /home/moon/yy/reid-animal/database/rp_kl
#python train.py --name rp_kl_efficient --use_efficient --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_kl
#python train.py --name rp_kl_hr --use_hr --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.01 --circle --data_dir /home/moon/yy/reid-animal/database/rp_kl
#python train.py --name rp_kl_dense --use_dense --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/rp_kl

#python train.py --name test --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/RedPanda-33
#python train.py --name test --use_convnext --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/RedPanda-33
#python train.py --name test --use_swinv2 --lr 0.03 --warm_epoch 5 --erasing_p 0.5 --circle --data_dir /home/moon/yy/reid-animal/database/RedPanda-33
#python train.py --name test --use_efficient --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/RedPanda-33
#python train.py --name test --use_hr --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.01 --circle --data_dir /home/moon/yy/reid-animal/database/RedPanda-33
#python train.py --name test --use_dense --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/RedPanda-33

#python train.py --name t0_resnet50 --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/syn/ipd
#python train.py --name t0_convnext --use_convnext --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/syn/ipd
#python train.py --name t0_swinv2 --use_swinv2 --lr 0.03 --warm_epoch 5 --erasing_p 0.5 --circle --data_dir /home/moon/yy/reid-animal/database/syn/ipd
#python train.py --name t0_efficient --use_efficient --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/syn/ipd
#python train.py --name t0_hr --use_hr --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.01 --circle --data_dir /home/moon/yy/reid-animal/database/syn/ipd
#python train.py --name t0_dense --use_dense --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/syn/ipd
#
#python train.py --name t1_resnet50 --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/syn/rpd
#python train.py --name t1_convnext --use_convnext --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/syn/rpd
#python train.py --name t1_swinv2 --use_swinv2 --lr 0.03 --warm_epoch 5 --erasing_p 0.5 --circle --data_dir /home/moon/yy/reid-animal/database/syn/rpd
#python train.py --name t1_efficient --use_efficient --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/syn/rpd
#python train.py --name t1_hr --use_hr --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.01 --circle --data_dir /home/moon/yy/reid-animal/database/syn/rpd
#python train.py --name t1_dense --use_dense --warm_epoch 5 --stride 1 --erasing_p 0.5 --lr 0.02 --circle --data_dir /home/moon/yy/reid-animal/database/syn/rpd
