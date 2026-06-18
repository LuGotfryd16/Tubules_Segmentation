# Tubules_Segmentation
Semantic segmentation model for seminiferous tubules in H&amp;E sections of mouse testis (Mus musculus, CF-1 strain). Segments 3 classes (0 = background, 1 = epithelium, 2 = lumen) and derives calibrated morphometric metrics. Architecture: EfficientNet-B4 encoder + UNet with a dual decoder (segmentation + boundary) and SCSE attention. 
