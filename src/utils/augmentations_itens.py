from pose.pre_processamento import  gaussian_blur, add_noise, flip_frame, translate_frame


AUGMENTATIONS = {
    'noise': add_noise,
    'translate': translate_frame,
    'flip': flip_frame,
    'gaussian': gaussian_blur
}
