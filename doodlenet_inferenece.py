import keras
import argparse
from keras.preprocessing import image
from blog_pretrained import preprocess
from keras.models import model_from_json
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

DOC_STRING = """Script for performing inference on an input image."""
parser = argparse.ArgumentParser(description=DOC_STRING)
parser.add_argument("-image", "--image", type=str, help="path to image (jpg) for classification")


try:
    json_file = open(MODELPATH, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
except:
    sys.exit("Can't find model at {MODELPATH}".format(MODELPATH=MODELPATH))

try:
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(WEIGHTSPATH)
except:
    sys.exit("Can't find model weights at {WEIGHTSPATH}".format(WEIGHTSPATH=WEIGHTSPATH))


if __name__ == "__main__":
    # Parse commandline arguments...
    args = parser.parse_args()

    # Echo command line...
    sys.stderr.write("%s\n" % " ".join(sys.argv))

    ### CONSTANTS!
    MODELPATH = 'doodlenet/longtrain_model.json'
    WEIGHTSPATH = 'doodlenet/longtrain_weights.h5'

    image = preprocess(path=args.image)
    prediction, confidence = loaded_model.predict([image][0]
    if int(prediction) == 1:
        print "This is a picture of Harrison. I'm this sure: {CONFIDENCE}".format(CONFIDENCE=confidence)
    elif int(prediction) == 0:
        print "This is not a picture of Harrison. I'm this sure: {CONFIDENCE}".format(CONFIDENCE=confidence)

    