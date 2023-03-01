# Deep-Fake-Detection-using-MesoNet


To detect deepfake photos, we decided to go ahead with the two state-of-the-art deep learningbased methods, namely Meso-4 and MesoInception-4. These networks are a part of the research
conducted by The University of California, Riverside and are discussed and implemented in the
paper MesoNet: a Compact Facial Video Forgery Detection Network by Darius Afchar et. Al. on
generic video datasets. Their proposed approach was originally tested against deepfakes using a
private database, achieving a 98.4% of fake detection accuracy for the best performance. A
mesoscopic level of analysis is used for the models, as at a microscopic level, image noise is
strongly degraded and at higher semantic levels, the human eye struggles to distinguish forged
images. Hence, an intermediate approach is employed using a deep neural network with a small
number of layers.

<img width="391" alt="image" src="https://user-images.githubusercontent.com/68967101/222268512-e892d61f-45e0-4560-9c21-fd5341ced786.png">

