# Pytorch-Facial-Expression-Recognition
This trained model recevies a colorful or grey image of an face and predicts emotion portays in it.

The availbe emotions are:
- Happy
- Sad
- Disgust
- Neutral
- Suprise
- Fear
- Angry

In training of this model these resources and technologies were used: 
- Pytorch framework
- EfficientB2 for Architecture
- [This Kaggle dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)
- AdamW as optimizer Algotrim
- CrossEntropyLoss as loss function
- Google colab & Kaggle
- And Gradio & Huggingface were utilised for Deployment


In my gradio implementation images are preprocessed to be "Nomalized", so colorful images will work as well on this model.

![image](https://github.com/Bijan-K/Pytorch-Facial-Expression-Recognition/assets/80640045/be2e5f5d-7cf3-41da-98cd-af55e0b4ad59)

You can go to the working huggingface space from this [link](https://huggingface.co/spaces/bijankn/Facial_Expression_Recognition).

