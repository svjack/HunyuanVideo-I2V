export PYTHONPATH=${PYTHONPATH}:`pwd`
for ((i=0;i<$HOST_GPU_NUM;++i)); do
    CUDA_VISIBLE_DEVICES=$i python3 -u video_latent_extract/run.py --local_rank $i --config 'video_latent_extract/vae.yaml'&
done
# CUDA_VISIBLE_DEVICES=0 python3 -u video_latent_extract/run.py --local_rank 0 --config 'video_latent_extract/vae.yaml'&
wait

echo "Finished."