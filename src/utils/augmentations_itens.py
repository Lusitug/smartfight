from pose.preprocessamento.augumentador_video import AugumentadorVideo

class AugmentationItens:
    AUGMENTATIONS = {
        'noise': AugumentadorVideo.aplicar_ruido,
        'translate': AugumentadorVideo.translate_frame,
        'shiny+': AugumentadorVideo.aumentar_brilho,
        'shiny-': AugumentadorVideo.reduzir_brilho,
        'gaussian': AugumentadorVideo.borrao_gaussian,
        'scale': AugumentadorVideo.scale
    }
