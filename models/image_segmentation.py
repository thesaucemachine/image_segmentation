"""
From: https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/segmentation.ipynb 
"""

import tensorflow as tf
import tensorflow_datasets as tfds # not needed
from tensorflow_examples.models.pix2pix import pix2pix
from IPython.display import clear_output
import matplotlib.pyplot as plt


def unet_model(output_channels:int):
    
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2,
        padding='same')  #64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

#In addition, the image color values are normalized to the [0,1] range. Finally, as mentioned above the pixels in the segmentation mask are labeled either {1, 2, 3}. 
#For the sake of convenience, subtract 1 from the segmentation mask, resulting in labels that are : {0, 1, 2}.

def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1

    return input_image, input_mask

def load_image(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))
    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

# The following class performs a simple augmentation by randomly-flipping an image.

class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        # both use the same seed, so they'll make the same random changes.
        self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
    
    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)

        return inputs, labels

def display(display_list):
    
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

def create_mask(pred_mask):
        pred_mask = tf.argmax(pred_mask, axis=-1)
        pred_mask = pred_mask[..., tf.newaxis]
        return pred_mask[0]

def show_predictions(dataset=None, num=1):
    
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask,
                create_mask(model.predict(sample_image[tf.newaxis, ...]))])

class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

def main():

    ## DL dataset

    dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

    #The dataset already contains the required training and test splits, so continue to use the same splits.

    TRAIN_LENGTH = info.splits['train'].num_examples
    BATCH_SIZE = 64
    BUFFER_SIZE = 1000
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

    train_images = dataset['train'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_images = dataset['test'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    #Build the input pipeline, applying the Augmentation after batching the inputs.

    train_batches = (
    train_images
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE))

    test_batches = test_images.batch(BATCH_SIZE)

    # Visualize an image example and its corresponding mask from the dataset.

    for images, masks in train_batches.take(2):
        sample_image, sample_mask = images[0], masks[0]
        display([sample_image, sample_mask])

    # define the model

    base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]

    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

    down_stack.trainable = False

    # The decoder/upsampler is simply a series of upsample blocks implemented in TensorFlow examples.

    up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
    ]

    # train the model

    OUTPUT_CLASSES = 3

    model = unet_model(output_channels=OUTPUT_CLASSES)

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    # look at the model

    tf.keras.utils.plot_model(model, show_shapes=True)

    # Try out the model to check what it predicts before training.

    show_predictions()

    # run model.fit

    EPOCHS = 20
    VAL_SUBSPLITS = 5
    VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

    model_history = model.fit(train_batches, epochs=EPOCHS,
                            steps_per_epoch=STEPS_PER_EPOCH,
                            validation_steps=VALIDATION_STEPS,
                            validation_data=test_batches,
                            callbacks=[DisplayCallback()])

    # show loss

    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    plt.figure()
    plt.plot(model_history.epoch, loss, 'r', label='Training loss')
    plt.plot(model_history.epoch, val_loss, 'bo', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()

    # make predictions

    show_predictions(test_batches, 3)

def add_sample_weights(image, label):
    # The weights for each class, with the constraint that:
    #     sum(class_weights) == 1.0
    class_weights = tf.constant([2.0, 2.0, 1.0])
    class_weights = class_weights/tf.reduce_sum(class_weights)

    # Create an image of `sample_weights` by using the label at each pixel as an 
    # index into the `class weights` .
    sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))

    return image, label, sample_weights

def look_at_class_imbalance():
    try:
        model_history = model.fit(train_batches, epochs=EPOCHS,
                                steps_per_epoch=STEPS_PER_EPOCH,
                                class_weight = {0:2.0, 1:2.0, 2:1.0})
        assert False
    except Exception as e:
        print(f"Expected {type(e).__name__}: {e}")

    label = [0,0]
    prediction = [[-3., 0], [-3, 0]] 
    sample_weight = [1, 10] 

    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                reduction=tf.losses.Reduction.NONE)
    loss(label, prediction, sample_weight).numpy()

    # look into class_weight list. The resulting dataset elements contain 3 images each
    train_batches.map(add_sample_weights).element_spec

    # now you can train a model on this weighted dataset

    weighted_model = unet_model(OUTPUT_CLASSES)
    weighted_model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    weighted_model.fit(train_batches.map(add_sample_weights),epochs=1,steps_per_epoch=10)

if __name__ == "__main__":

   
    main() # make unet and image segmentation stuff

    look_at_class_imbalance() # look at class imabalance stuff