- Pre-training script:
    ```Python
    ./run.sh recipes/is2022/pretrain.py \
    --identifier vqvae_pretrain_0.00f0 \
    --train-configs recipes/is2022/configs/train/pretrain.yml \
    --model-configs recipes/is2022/configs/model/ver2/vqvae/pretrain_0.00f0.yml 
    ```
- Denoising script:
    ```Python
    ./run.sh recipes/is2022/denoise.py \
    --identifier vqvae_denoise_0.00f0 \
    --train-configs recipes/is2022/configs/train/denoise.yml \
    --model-configs recipes/is2022/configs/model/ver2/vqvae/denoise_qe_0.02f0_wiener.yml \
    --pretrain-model-configs recipes/is2022/configs/train/pretrain.yml \
    --pretrain-model-path <path-to-pre-trained-model-weight>
    ```