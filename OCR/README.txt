Requirements:

TensorFlow=2.1.0 

Command Line Arguments:
"-tr <str>"    -> training data,   default='emnist-byclass-train.csv'
"-te <str>"    -> test data,   default='emnist-byclass-test.csv'
"-cl <int>" -> classes, default=62
"-bs <int>"    -> batch size,       default=128
"-lr <float>"  -> learningrate,    default=0.001
"-e <int>" -> epochs, default=10
"-im <str>" -> input images file, default='input_images'
"-o <str>" -> output file name, default='translations.txxt'
"-bs <int>"    -> batch size,       default=128

example: python main.py -e 5 -lr 0.01 -bs 128


Explanation:
The code is originally trained using tensorflow-gpu for computational convenience.
Since the training and test data is large in terms of storage, I could not put it under the zip file.
You can download the dataset from here: https://www.kaggle.com/crawford/emnist
Or through this drive: https://drive.google.com/drive/folders/11EOzbxu17PsLSQp2mKoZX6m3p_LnOKzO?usp=sharing
The training takes about minutes for 15 epochs under current setup.
For the images input folder, I have included a sample which consists of the images from the original github project, as well as new ones I've added.
You can also check a handwritten image and check its translation, if you add anything to the "images" folder.

There exists an alternative code where the model is pretrained. You can check the pretrained folder in order to skip the training process and use the model trained before.
The instructions for the pretrained code is inside the pretrained folder.

