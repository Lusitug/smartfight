from pose.pre_processamento import  aplicar_ruido, borrao_gaussian, aumentar_brilho, reduzir_brilho, translate_frame,scale

AUGMENTATIONS = {
    'noise': aplicar_ruido,
    'translate': translate_frame,
    'shiny+': aumentar_brilho,
    'shiny-': reduzir_brilho,
    'gaussian': borrao_gaussian,
    'sacel': scale
}
