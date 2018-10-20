import sys
import glob
import keras
import os.path
import argparse
import numpy as np
import tensorflow as tf
from time import time
from keras import optimizers
from keras import backend as K
from keras.applications import vgg16, inception_resnet_v2
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.engine import Model
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Conv2D
from keras.callbacks import TensorBoard
from numpy.random import seed
from tensorflow import set_random_seed

### SET SEED!
SEED = 8
seed(SEED)
set_random_seed(SEED)


DOC_STRING = """ 
Example training run:
    python2.7 blog_pretrained.py \
    --method notop \
    --model vgg \
    --train_dir data/train/ \
    --test_dir data/validation/ \
    --model_json longtrain_model.json \
    --out_weights longtrain_weights.h5 \
    --epochs 20 \
    --batch_size 20 \
    --optimizer SGD \
    --LR 0.0001 \
    --nesterov \
    --hidden 200 \
    --class_mode categorical
"""
parser = argparse.ArgumentParser(description=DOC_STRING)
parser.add_argument("-image", "--image", type=str, help="path to image for classification")
parser.add_argument("-dir", "--image_dir", type=str, help="path to directory of images for classification")
parser.add_argument("-method", "--method", type=str, default="1", help="Method to use: [vgg|notop]")
parser.add_argument("-train", "--train_dir", type=str, help="path to directory of images for training")
parser.add_argument("-test", "--test_dir", type=str, help="path to directory of images for testing")
parser.add_argument("-epochs", "--epochs", type=int, help="Number of epochs for training")
parser.add_argument("-bs", "--batch_size", type=int, help="Batch size")
parser.add_argument("-hidden", "--hidden", type=int, default=512, help="Size of hidden layers")
parser.add_argument('-model_json', '--model_json', type=str, default="model.json", help="Path to output model")
parser.add_argument('-out_weights', '--out_weights', type=str, default="weights.h5", help="Path to h5 file to save weights")
parser.add_argument("-LR", "--LR", type=float, help="Learning rate for optimizer")
parser.add_argument("-momentum", "--momentum", type=float, default=0.9, help="Specify SGD momentum")
parser.add_argument("-decay", "--decay", type=float, default=0.0, help="Specify SGD decay")
parser.add_argument("-nesterov", "--nesterov", action='store_true', help="Use nesterov momentum")
parser.add_argument("-optimizer", "--optimizer", type=str, help="Specify optimizer [SGD|RMS|Adam]")
parser.add_argument("-model", "--model", type=str, help="Specify the model [vgg|irv2]")
parser.add_argument("-class_mode", "--class_mode", type=str, help="Specify class mode [binary|categorical]")

def preprocess(path):
    # load an image in PIL format
    original = load_img(path, target_size=(224, 224))
    if original.size != (224, 224):
        #print original.size, "....@..."
        original = original.resize((224, 224))
    # convert the PIL image to a numpy array
    # IN PIL - image is in (width, height, channel)
    # In Numpy - image is in (height, width, channel)
    numpy_image = img_to_array(original)
    
    # Convert the image / images into batch format
    # expand_dims will add an extra dimension to the data at a particular axis
    # We want the input matrix to the network to be of the form (batchsize, height, width, channels)
    # Thus we add the extra dimension to the axis 0.
    image_batch = np.expand_dims(numpy_image, axis=0)
    
    # prepare the image for the VGG model...
    processed_image = vgg16.preprocess_input(image_batch.copy())
    return processed_image

if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "136"

    args = parser.parse_args()

    # Echo command line...
    sys.stderr.write("%s\n" % " ".join(sys.argv))

    # Load image(s)...
    if args.image_dir:
        IMAGES = glob.glob(os.path.join(args.image_dir, '*.jpg'))

    elif args.image:
        IMAGES = [args.image]

    if args.method == "vgg":
        #Load the VGG model
        vgg_model = vgg16.VGG16(weights='imagenet')
        
        # Loop over images...
        for image in IMAGES:
            processed_image = preprocess(path=image)
            predictions = vgg_model.predict(processed_image)
            label = decode_predictions(predictions)
            
            # Print output...
            print image
            for _, cl, pr in label[0]:
                print "\t", cl, ":\t", pr
            print "---------"
    print args
    if args.method == "notop":
        K.set_session(K.tf.Session(
            config=K.tf.ConfigProto(
                intra_op_parallelism_threads=136, 
                inter_op_parallelism_threads=1)
            )
        )  

        # Set parameters...
        NCL = 2
        H = args.hidden
        BS = args.batch_size
        IM_W = 224
        IM_H = 224
        
        # Image augmentation...
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.3,
            height_shift_range=0.3,
            # shear_range=0.2,
            zoom_range=0.3,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        test_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.3,
            height_shift_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        train_generator = train_datagen.flow_from_directory(
            args.train_dir,
            target_size=(IM_H, IM_W),
            batch_size=BS,
            class_mode=args.class_mode
        )
        sys.stderr.write("Getting training data from {DIR}...\n".format(DIR=args.train_dir))
        
        # Validation...
        validation_generator = test_datagen.flow_from_directory(
            args.test_dir,
            target_size=(IM_H, IM_W),
            batch_size=BS,
            class_mode=args.class_mode
        )
        
        #Load the VGG model...
        if args.class_mode == 'binary':
            out_units = 1
        elif args.class_mode == 'categorical':
            out_units = 2
        if args.model == "vgg":
            model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(IM_H, IM_W, 3))
            for layer in model.layers:
                layer.trainable = False
            
            last_layer = model.get_layer(model.layers[-1].name).output
            x = Flatten(name='flatten')(last_layer)
            x = Dense(H, activation='relu', name='fc6')(x)
            x = Dense(H, activation='relu', name='fc7')(x)
            out = Dense(units=out_units, activation='softmax', name='fc8')(x)
            custom_model = Model(model.input, out)
            
        if args.optimizer == "SGD":
            opt = optimizers.SGD(lr=args.LR, decay=args.decay, momentum=args.momentum, nesterov=args.nesterov)
            sys.stderr.write("Using SGD optimzer with learning rate {LR}, decay {DECAY}, momentum {MOMENTUM}, and nesterov set to {NESTEROV}...\n".format(LR=args.LR, DECAY=args.decay, MOMENTUM=args.momentum, NESTEROV=args.nesterov))
        elif args.optimizer == "RMS":
            opt = optimizers.RMSprop(lr=args.LR)
            sys.stderr.write("Using RMSprop optimzer with learning rate {LR}...\n".format(LR=args.LR))
        elif args.optimizer == "Adam":
            opt = optimizers.RMSprop(lr=args.LR)
            sys.stderr.write("Using RMSprop optimzer with learning rate {LR}...\n".format(LR=args.LR))
        custom_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    
    
    custom_model.fit_generator(
        train_generator, 
        # steps_per_epoch=len(train_generator), 
        steps_per_epoch = 921 // args.batch_size,
        epochs=args.epochs, 
        validation_data=validation_generator, 
        validation_steps=1,
        callbacks=[tensorboard],
    )
    
    # Save model...
    model_json = custom_model.to_json()
    with open(args.model_json, 'w') as json_file:
        json_file.write(model_json)
    sys.stderr.write("Saved model to {PATH}...\n".format(PATH=args.model_json))

    # Save weights...
    custom_model.save_weights(args.out_weights)
    sys.stderr.write("Saved weights to {PATH}...\n".format(PATH=args.out_weights))