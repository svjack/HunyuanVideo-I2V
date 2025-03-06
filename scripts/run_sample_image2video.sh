#!/bin/bash

python3 sample_image2video.py \
    --prompt "A man with short gray hair plays a red electric guitar." \
    --i2v-image-path ./assets/demo/i2v/imgs/0.png \
    --model HYVideo-T/2 \
    --i2v-mode \
    --i2v-resolution 720p \
    --infer-steps 50 \
    --video-length 129 \
    --flow-reverse \
    --flow-shift 17.0 \
    --seed 0 \
    --use-cpu-offload \
    --save-path ./results \

# More examples
#    --prompt "A woman sits on a wooden floor, holding a colorful bag." \
#    --i2v-image-path ./assets/demo/i2v/imgs/1.png \
#
#    --prompt "A woman with long blonde braids speaks while gesturing with her hands." \
#    --i2v-image-path ./assets/demo/i2v/imgs/2.png \
#
#    --prompt "A transparent sphere with gold embellishments rotates against a blue background." \
#    --i2v-image-path ./assets/demo/i2v/imgs/3.png \
#
#    --prompt "The cat grabs a ball and plays with it." \
#    --i2v-image-path ./assets/demo/i2v/imgs/4.png \