

Command Line Arguments:
"-te <str>"    -> test data,   default='emnist-byclass-test.csv'
"-im <str>" -> input images file, default='input_images'
"-o <str>" -> output file name, default='translations.txt'

example: python pretrained.py -im 


Explanation:
The pretrained code has the pretrained model, thus allowing computational easiness for users.
"model.json" and "model.h5" files should be present under the directory.
You can download the dataset from here: https://www.kaggle.com/crawford/emnist
The training takes about minutes for 15 epochs under current setup. The code needs only the test csv file in order to run.
For the images input folder, I have included a sample which consists of the images from the original github project, as well as new ones I've added.
You can also check a handwritten image and check its translation, if you add anything to the "images" folder.

