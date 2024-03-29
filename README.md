Project Description

Brain tumour segmentation is one of the most challenging problems in medical image analysis. Traditional deep learning models, notably the U-Net architecture, show promise for image segmentation tasks but are underutilised in the context of brain tumour segmentation due to limited annotated data. However, MRI images categorised by tumour type are more accessible.

This project proposes a novel approach leveraging the encoder-decoder architecture of U-Net for brain tumour segmentation. By first employing the model for tumour classification, the encoder learns critical features to be an efficient feature extractor for subsequent image segmentation tasks. Essentially, the classification task acts as a pre-training phase, enabling the model to develop a nuanced understanding of brain MRI features, which is subsequently leveraged in the more complex task of segmentation.

In the phase of transferring learnt brain features on the segmentation task, encoder part of the proposed U-Net would be frozen and only practice training on the U-Net decoder.
