
# alphas=(10 1 0.1 0.01)
# alphas=(3 0.3 0.1 0.03 0.01)
# betas=(3 0.3 0.03 0.003)
seeds=(10 11 12 13 14 15 16 17 18)
label_rates=(0.01)

sigmas=(100)
alphas=(3)
for sigma in ${sigmas[@]};
do 
    for alpha in ${alphas[@]};
    do
        for seed in ${seeds[@]};
        do
            python train_RSGNN.py \
                --seed ${seed} \
                --dataset cora \
                --attack meta \
                --ptb_rate 0.0  \
                --alpha ${alpha} \
                --sigma ${sigma}  \
                --beta 0.3 \
                --t_small 0.05 \
                --lr  1e-3 \
                --lr_adj  1e-3\
                --epoch 1000 \
                --label_rate 0.01 \
                --threshold 0.8 \
                --n_p=100
        done
    done
done

alphas=(0.01)
for sigma in ${sigmas[@]};
do 
    for alpha in ${alphas[@]};
    do
        for seed in ${seeds[@]};
        do
            python train_RSGNN.py \
                --seed ${seed} \
                --dataset cora \
                --attack meta \
                --ptb_rate 0.15  \
                --alpha ${alpha} \
                --sigma ${sigma}  \
                --beta 0.3 \
                --t_small 0.1 \
                --lr  1e-3 \
                --lr_adj  1e-3\
                --epoch 1000 \
                --label_rate 0.01 \
                --threshold 0.8 \
                --n_p=100
        done
    done
done
alphas=(0.3)
for sigma in ${sigmas[@]};
do 
    for alpha in ${alphas[@]};
    do
        for seed in ${seeds[@]};
        do
            python train_RSGNN.py \
                --seed ${seed} \
                --dataset cora \
                --attack nettack \
                --ptb_rate 0.15  \
                --alpha ${alpha} \
                --sigma ${sigma}  \
                --beta 0.03 \
                --t_small 0.0 \
                --lr  1e-3 \
                --lr_adj  1e-3\
                --epoch 1000 \
                --label_rate 0.01 \
                --threshold 0.8 \
                --n_p=100
        done
    done
done
alphas=(3)
for sigma in ${sigmas[@]};
do 
    for alpha in ${alphas[@]};
    do
        for seed in ${seeds[@]};
        do
            python train_RSGNN.py \
                --seed ${seed} \
                --dataset cora \
                --attack random \
                --ptb_rate 0.3  \
                --alpha ${alpha} \
                --sigma ${sigma}  \
                --beta 0.3 \
                --t_small 0.1 \
                --lr  1e-3 \
                --lr_adj  1e-3\
                --epoch 1000 \
                --label_rate 0.01 \
                --threshold 0.8 \
                --n_p=100 
        done
    done
done