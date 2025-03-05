#!/bin/bash

# I2V
python3 sample_image2video.py \
    --prompt "A man with short gray hair plays a red electric guitar." \
    --i2v-image-path ./assets/demo/i2v/imgs/0.png \
    --save-path ./results \
    --i2v-mode \
    --i2v-resolution 720p \
    --model HYVideo-T/2 \
    --cfg-scale 1.0 \
    --infer-steps 50 \
    --video-length 129 \
    --flow-reverse \
    --seed 0 \
    --use-cpu-offload \
    --flow-shift 17.0 \

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

# I2V Lora
#python3 sample_image2video.py \
#    --lora-path ./ckpts/lora/embrace_kohaya_weights.safetensors \
#    --prompt "Two people hugged tightly, In the video, two people are standing apart from each other. They then move closer to each other and begin to hug tightly. The hug is very affectionate, with the two people holding each other tightly and looking into each other's eyes. The interaction is very emotional and heartwarming, with the two people expressing their love and affection for each other." \
#    --i2v-image-path ./assets/demo/i2v_lora/imgs/embrace.png \
#    --save-path ./results \
#    --i2v-mode \
#    --i2v-resolution 540p \
#    --model HYVideo-T/2 \
#    --cfg-scale 1.0 \
#    --infer-steps 50 \
#    --video-length 129 \
#    --flow-reverse \
#    --seed 0 \
#    --use-cpu-offload \
#    --flow-shift 5.0 \
#    --use-lora \
#    --lora-scale 1.0 \

#    --lora-path ./ckpts/lora/hair_replacement_kohaya_weights.safetensors \
#    --prompt "rapid_hair_growth, The hair of the characters in the video is growing rapidly. The character's hair undergoes a dramatic transformation, growing rapidly from a short, straight style to a long, wavy one. Initially, the hair is a light blonde color, but as it grows, it becomes darker and more voluminous. The character's facial features remain consistent throughout the transformation, with a slight change in the shape of the jawline as the hair grows. The clothing changes from a simple, casual outfit to a more elaborate, fashionable ensemble that complements the longer hair. The overall appearance shifts from a casual, everyday look to a more stylish, sophisticated one. The character's expression remains calm and composed throughout the transformation, with a slight smile as the hair grows." \
#    --i2v-image-path ./assets/demo/i2v_lora/imgs/hair_replacement.png \

