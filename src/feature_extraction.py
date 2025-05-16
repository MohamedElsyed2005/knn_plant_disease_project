import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
from joblib import Parallel, delayed
import multiprocessing
from skimage.feature import hog
from sklearn.decomposition import PCA
import albumentations as A
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array



transform = A.Compose([
    A.HorizontalFlip(p=0.5), # 50% chance of horizontal flip
    A.VerticalFlip(p=0.2), # 20% chance of vertical flip
    A.RandomBrightnessContrast(p=0.3), # 30% chance of adjusting brightness/contrast
    A.ShiftScaleRotate(
        shift_limit=0.05, # Shift by 5% in any direction
        scale_limit=0.1,  # Scale up or down by 10%
        rotate_limit=15,  # Rotate by ±15 degrees
        p=0.5),           # 50% chance of applying this
])


def augment_image(image):
    """
    This line applies a set of predefined augmentations (random image transformations)
    to the input image using Albumentations, a powerful image augmentation library.
    
    The transform object is created using A.Compose([]), which means it is a list of multiple transformations.

    The result of applying this transformation is a dictionary that looks like this:
    >>> augmented = transform(image=image)
    augmented = {
        'image': <Augmented_Image_Array>,
        'other_keys_if_present': ...
    }
    >>> augmented['image']
    to get the <Augmented_Image_Array>
    example: 
    [[ 1, 2, 3 ],
    [ 4, 5, 6 ],
    [ 7, 8, 9 ]]

    flip horizontal:      |   flip vertical   
        [[ 3, 2, 1 ],     |     [[ 7, 8, 9 ],
        [ 6, 5, 4 ],      |     [ 4, 5, 6 ],
        [ 9, 8, 7 ]]      |     [ 1, 2, 3 ]]

    """
    augmented = transform(image=image)
    return augmented['image']

def preprocess_image(image, target_size=(96, 96)):
    if image is None:
        print("Error: Image not loaded properly.")
        return None
    image = cv2.resize(image, target_size) # resize from (255,255,3) into (96, 96)
    """
    You asked:
    "How is it even possible to resize a (250, 250, 3) image down to (96, 96)?"
    (Without breaking or damaging the image?)

    The program does NOT cut or remove parts of the image.
    Instead, it uses a mathematical method called interpolation to smartly "resize" the content.

    What is Interpolation?
    Simply:
    - Instead of just deleting pixels, interpolation calculates new pixel values.
    - It uses nearby pixels to guess how the new pixels should look.

    | Interpolation Method     | OpenCV Flag         | Best For                   | Speed       | Quality      | Notes                                 |
    |--------------------------|---------------------|----------------------------|------------ |--------------|---------------------------------------|
    |   Nearest Neighbor       |  cv2.INTER_NEAREST  | Binary masks, segmentation | **Fastest** | **Lowest**   | May look blocky, no smoothing         |
    |   Bilinear (Default)     |  cv2.INTER_LINEAR   | General use Upscaling      | Fast        | Good         | Default method, smooth results        |
    |   Bicubic                |  cv2.INTER_CUBIC    | Upscaling images           | Slower      | Better       | Sharper than bilinear                 |
    |   Area-based             |  cv2.INTER_AREA     |   Downscaling              | Fast        | Very Good    | Maintains image quality when shrinking|
    |   Lanczos                |  cv2.INTER_LANCZOS4 | High-quality upscaling     | Slowest     | Best         | For fine detail, computationally heavy|
    ----------------------------------------------------------------------------------------------------------------------------------------------------
    
    default 
    >>> image = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)
    """ 
    # convert image from RGB (blue, green, red) into LAB (Lightness, A: green to red, B: blue to yellow)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)                                  # split into L, A, B
    # CLAHE (Contrast Limited Adaptive Histogram Equalization).
    # It improves brightness in dark and bright areas without losing details.
    # clipLimit=2.0: Sets a maximum contrast limit to avoid over-enhancement.
    # tileGridSize=(8,8): Defines the size of the grid for contrast enhancement (8x8 means the image is divided into small tiles).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) 
    l = clahe.apply(l)                                        # Apply CLAHE only on the Lightness channel (L)
    lab = cv2.merge([l, a, b])                               # Merge the three channels back after enhancement
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)          # Convert the enhanced LAB image back to BGR
    return enhanced

# Function to extract color features from an image
def extract_color_features(image):
    features = []
    # Convert the image to HSV (Hue, Saturation, Value) and LAB color spaces
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Loop through both color spaces (HSV and LAB)
    for color_space in [hsv, lab]:
        # Split the color space into its channels
        for channel in cv2.split(color_space):
            # Calculate the mean and standard deviation of each channel
            mean = np.mean(channel)
            std = np.std(channel)
            features.extend([mean, std])

    # Return color features as a flattened NumPy array
    return np.array(features, dtype=np.float32)

# Function to extract texture features using HOG (Histogram of Oriented Gradients)
def extract_texture_features(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Extract HOG features from the grayscale image
    hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), visualize=False)

    # Return HOG features
    return hog_features

# Function to extract disease-specific features (infected ratio) from the image
def extract_disease_features(image):
    # Convert the image to HSV and use the V (brightness) channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)

    # Calculate the ratio of dark pixels (potentially infected areas)
    infected_ratio = np.mean(v < 100)

    # Return this value as a single-element NumPy array
    return np.array([infected_ratio], dtype=np.float32)

# Load a pre-trained MobileNetV2 model (without the top classification layer)
mobilenet_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(96, 96, 3))

# Function to extract deep learning features using MobileNetV2
def extract_deep_features(image):
    # Convert the image to RGB format and resize to MobileNetV2's input size
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (96, 96))

    # Convert the image to an array and preprocess it for MobileNetV2
    x = img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Extract deep features using MobileNetV2
    features = mobilenet_model.predict(x)

    # Flatten the feature map to a one-dimensional array
    return features.flatten()

# Master function to extract all features (color, texture, disease, deep features)
def extract_features(image):
    color_features = extract_color_features(image)
    texture_features = extract_texture_features(image)
    disease_features = extract_disease_features(image)
    deep_features = extract_deep_features(image)

    # Concatenate all features into one vector
    return np.concatenate([color_features, texture_features, disease_features, deep_features])

def process_single_image(img_path, target_size=(96, 96)):
    image = cv2.imread(img_path) # laod images
    if image is None:
        print(f"Warning: Could not load image {img_path}. Skipping...")
        return None, None

    image = augment_image(image)  # apply augmentations

    processed_image = preprocess_image(image, target_size) # apply preprocessing
    if processed_image is None:
        return None, None
    features = extract_features(processed_image)
    return features, os.path.basename(os.path.dirname(img_path))

def extract_features_from_folder(folder_path, target_size=(96, 96)):
    """
    the structure of data 
        data/
            |
            ├──PlantVillage/
                    ├── Pepper__bell___Bacterial_spot
                    ├── Pepper__bell___healthy
                    ├── Potato___Early_blight
                    ├── Potato___Early_blight
                    ├── Potato___healthy
                    ├── Potato___Late_blight
                    ├── Tomato__Target_Spot
                    ├── Tomato__Tomato_mosaic_virus
                    ├── Tomato__Tomato_YellowLeaf__Curl_Virus
                    ├── Tomato_Bacterial_spot
                    ├── Tomato_Early_blight
                    ├── Tomato_healthy
                    ├── Tomato_Late_blight
                    ├── Tomato_Leaf_Mold
                    ├── Tomato_Septoria_leaf_spot
                    ├── Tomato_Spider_mites_Two_spotted_spider_mite
    --------------------------------------------------------------------------------------------------
    to read the data from each folder. we need to os library to deal with folders and files 
        >>> import os 
        
        
    >>> os.listdir ==> to get all folders names
    Example:
        >>> os.listdir(folder_path) 
        >>> ["Pepper__bell___Bacterial_spot","Pepper__bell___healthy", "Potato___Early_blight", ...]

    >>> os.path.join ==> combine the folder path with inner folder to get their path
    Example:
        >>> folder_path = "..//data//PlantVillage"
        ... label_name = "Pepper__bell___Bacterial_spot"
        ... label_folder = os.path.join(folder_path, label_name)
        >>> label_folder
        ..//data//PlantVillage//Pepper__bell___Bacterial_spot
    
    >>> os.listdir(label_folder)
    Example: 
        >>> label_folder = "..//data//PlantVillage//Pepper__bell___Bacterial_spot"
        ... os.listdir(label_folder)
        ["0a0dbf1f-1131-496f-b337-169ec6693e6f___NREC_B.Spot 9241.JPG", "0a4c007d-41ab-4659-99cb-8a4ae4d07a55___NREC_B.Spot 1954.JPG", ...]
        >>> all_paths.append(os.path.join(label_folder, img_name))
        ["..//data//PlantVillage//Pepper__bell___Bacterial_spot//0a0dbf1f-1131-496f-b337-169ec6693e6f___NREC_B.Spot 9241.JPG", ...]
    ------------------------------------------------------------------------------------------------------------
    """
    all_paths = []
    for label_name in os.listdir(folder_path):
        label_folder = os.path.join(folder_path, label_name)
        if not os.path.isdir(label_folder): # check if the path is already exist or not
            continue
        for img_name in os.listdir(label_folder):
            all_paths.append(os.path.join(label_folder, img_name))

    print(f"Processing {len(all_paths)} images...")

    results = Parallel(n_jobs=multiprocessing.cpu_count() - 1)(
        delayed(process_single_image)(path, target_size) for path in all_paths)
    
    """
    What is Parallel Processing?
        - Parallel processing is a technique where a task is divided into smaller tasks, which are executed simultaneously on multiple CPU cores.
        - This can significantly speed up your code, especially when working with large datasets (like images here).
    
    joblib.Parallel is a utility from the joblib library that makes it very easy to parallelize tasks.
    It creates multiple parallel workers (processes) to perform tasks faster

    Syntax Explanation:
    >>> Parallel(n_jobs=-1)(delayed(func)(arg1, arg2) for arg1, arg2 in data) # default 
    ... n_jobs: Number of parallel workers (processes or threads).
        ... -1 means using all available CPU cores.
        ... multiprocessing.cpu_count() - 1 means using all CPU cores minus one (keeping one core free to avoid system overload).
    ... delayed: A utility that makes a function lazy (only runs when called).
    ... func: The function you want to run in parallel (process_single_image here).
    ... arg1, arg2: Arguments passed to the function.

    ----------------------------------------------------------------------------------------------------------------------------------------------
    |              without Parallel                                      |        With Parallel (7 Cores)
    ----------------------------------------------------------------------------------------------------------------------------------------------
    | >>> for path in all_paths:                                         | >>> results = Parallel(n_jobs=7)
    | >>> result = process_single_image(path, target_size)               | ...        (delayed(process_single_image)(path, target_size) for path in all_paths)
    | >>> results.append(result)                                         |
    ----------------------------------------------------------------------------------------------------------------------------------------------
    | If you have 20639 images and each takes 1 second:                  | 7 images are processed every second.
    | Total time = 20639 seconds (about 344 minutes = 5 h 44 min).       | Total time ≈ 20639  / 7 ≈ 49  min.
    ----------------------------------------------------------------------------------------------------------------------------------------------
    >>> results is a list of the outputs from process_single_image:
    [(features1, label1), (features2, label2), ...]
    """
    
    #            res[0]   res[1]
    # result = (features1, label1)      
    features = [res[0] for res in results if res[0] is not None] 
    labels = [res[1] for res in results if res[0] is not None]

    le = LabelEncoder() 
    # ["Pepper__bell___Bacterial_spot","Pepper__bell___healthy", "Potato___Early_blight", ...]
    # convert them into [ 0, 1, 2, ...]
    encoded_labels = le.fit_transform(labels) 

    # Stacking them vertically using np.vstack
    features = np.vstack(features)
    # apply dimensionality reduction using PCA
    pca = PCA(n_components=min(300, features.shape[1])) # features will be 300 cols
    features_reduced = pca.fit_transform(features)

    print("Features extracted and reduced.")
    return features_reduced, encoded_labels, le, pca

if __name__ == "__main__":
    print("Starting feature extraction...")
    features, labels, label_encoder, pca = extract_features_from_folder(r'../data/PlantVillage')

    os.makedirs('../saved_data', exist_ok=True)

    print("Saving data...")
    np.save('../saved_data/features.npy', features)
    np.save('../saved_data/labels.npy', labels)
    joblib.dump(label_encoder, '../saved_data/label_encoder.joblib')
    joblib.dump(pca, '../saved_data/pca_model.joblib')

    print("Data saved successfully.")