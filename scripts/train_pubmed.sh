
# alphas=(10 1 0.1 0.01)1
# alphas=(30 3 0.3 0.03)
# gammas=(3 0.3 0.03 0.003)
# ptb_rates=(0.1 0.2 0.3 0.4 0.5)
seeds=(10 11 12 13 14 15)
sigmas=(100)
for sigma in ${sigmas[@]};
do 
    for seed in ${seeds[@]};
    do
        python train_RSGNN.py \
            --seed ${seed} \
            --dataset pubmed \
            --attack meta \
            --ptb_rate 0.0  \
            --alpha 0.001 \
            --sigma ${sigma}  \
            --gamma 0.1 \
            --t_small 0.0 \
            --lr  1e-2 \
            --lr_adj  1e-2\
            --epoch 600 \
            --label_rate 0.1 \
            --threshold 0.8 \
            --n_p=0
    done
done

for sigma in ${sigmas[@]};
do 
    for seed in ${seeds[@]};
    do
        python train_RSGNN.py \
            --seed ${seed} \
            --dataset pubmed \
            --attack meta \
            --ptb_rate 0.15  \
            --alpha 0.001 \
            --sigma ${sigma}  \
            --gamma 0.1 \
            --t_small 0.0 \
            --lr  1e-2 \
            --lr_adj  1e-2\
            --epoch 600 \
            --label_rate 0.1 \
            --threshold 0.8 \
            --n_p=0
    done
done

for sigma in ${sigmas[@]};
do 
    for seed in ${seeds[@]};
    do
        python train_RSGNN.py \
            --seed ${seed} \
            --dataset pubmed \
            --attack nettack \
            --ptb_rate 0.15  \
            --alpha 0.001 \
            --sigma ${sigma}  \
            --gamma 0.1 \
            --t_small 0.0 \
            --lr  1e-2 \
            --lr_adj  1e-2\
            --epoch 600 \
            --label_rate 0.1 \
            --threshold 0.8 \
            --n_p=0
    done
done

for sigma in ${sigmas[@]};
do 
    for seed in ${seeds[@]};
    do
        python train_RSGNN.py \
            --seed ${seed} \
            --dataset pubmed \
            --attack random \
            --ptb_rate 0.3  \
            --alpha 0.001 \
            --sigma ${sigma}  \
            --gamma 0.1 \
            --t_small 0.0 \
            --lr  1e-2 \
            --lr_adj  1e-2\
            --epoch 600 \
            --label_rate 0.1 \
            --threshold 0.8 \
            --n_p=0
    done
done

