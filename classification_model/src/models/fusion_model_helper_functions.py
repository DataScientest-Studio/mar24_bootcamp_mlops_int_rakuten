import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


@tf.function
def load(file_path, img_size=(250,250)):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, size=img_size)
    return img



def tokenize(text, tokenizer, max_sequence_length):

    train_sequences = tokenizer.texts_to_sequences(text)
    train_padded_sequences = pad_sequences(
        train_sequences,
        maxlen = max_sequence_length,
        padding="post",
        truncating="post",
    )

    return train_padded_sequences

    




def df_to_dataset(dataframe, target_col, features: list, batch_size, tokenizer, max_sequence_length, shuffle=True, seed=0, img_size=(250,250)):
    df = dataframe.copy()
    labels = df.pop(target_col)
    
    # Create a dataset with file paths and corresponding labels
    feature2 = df[features[1]].values

    # Convert the other features to numpy arrays
    feature1 = df[features[0]].values
    feature1 = tokenize(feature1, tokenizer, max_sequence_length)

    # Separate features and labels into two tuples
    ds = tf.data.Dataset.from_tensor_slices((feature1,feature2, labels)).map(lambda f1, f2, lb: ((f1, load(f2, img_size=img_size)), lb))
    

    
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe), seed=seed)
    
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds




def preprocess_data(string_tensor, image_tensor, integer_tensor):
    # Your processing logic here
    # For example, you might convert the string to numerical data
    string_processed = string_tensor 
    # Normalize the image tensor
    image_processed = image_tensor
    return (string_processed, image_processed), integer_tensor