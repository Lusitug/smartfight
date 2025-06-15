from pose.preprocessamento.augumentador_video import AugumentadorVideo

class AugmentationItens:
    AUGMENTATIONS = {
        'noise': AugumentadorVideo.aplicar_ruido,
        'translate': AugumentadorVideo.translate_frame,
        'gaussian': AugumentadorVideo.borrao_gaussian,
        "flip_h": AugumentadorVideo.flip_h,
        "flip_v": AugumentadorVideo.flip_v,
        # 'shiny+': AugumentadorVideo.aumentar_brilho,
        # 'shiny-': AugumentadorVideo.reduzir_brilho,
        # 'scale': AugumentadorVideo.scale,
    }
