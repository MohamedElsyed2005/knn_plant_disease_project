import os # deals with folder and files
import cv2 # OpenCV to read images
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# LBP Settings
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS
LBP_METHOD = "uniform"

def extract_lbp_features(image, numPoints=LBP_POINTS, radius=LBP_RADIUS):
    """
    This function extracts texture features from a particular image using a technique called Local Binary Pattern (LBP),
    an effective way to recognize patterns in images, such as the texture of the affected paper in the disease classification project.
        
        Example
        [[  90   110  120 ]
        [   85   100   95 ]   =  center = 100
        [  130   105   80 ]]

        1- compare each neighbor with center:
            if the neighbor > center => 1
            else => 0 
            
        [[ 0   1   1 ]
        [  0       0 ]
        [  1   1   0 ]]

        Read with the hands of the clock
        --> 01100110
        convert  into decimal = 102 and this the LBP value

    """
    # Check and convert data type if necessary
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    
    # We convert the image to gray (Grayscale) because LBP only works on exposure.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    # We select a central pixel and around it a set of pixels (their number is determined by numPoints = 8, radius = 1,and method = LBP_METHOD = 'uniform').
    # compare each neighbor with center:
    #        if the neighbor > center => 1
    #        else => 0 
    lbp = local_binary_pattern(gray, numPoints, radius, method=LBP_METHOD)
    # Converting an LBP image to a Histogram (frequency chart) represents the number of times each LBP value is repeated.
    (hist, _) = np.histogram(lbp.ravel(),          # single out the two-dimensional matrix LBP into a straight line (1D array).
                             bins=np.arange(0, numPoints + 3), # bins=np.arange(0, (8+3) = 11) ==> [0,1,2,3,4,5,6,7,8,9,10]
                             range=(0, numPoints + 2)) # This is the full range of values that can appear in LBP.
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    """
    exmple:
        >>> lbp = [[0, 1, 1],
                  [2, 2, 1],
                  [0, 9, 8]]
        >>> (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, numPoints + 3), range=(0, numPoints + 2))
        >>> lbp.revel()
            [0, 1, 1, 2, 2, 1, 0, 9, 8]
        >>> bins = [0,1,2,3,4,5,6,7,8,9,10]

        | values | Number of impressions|
        |--------|----------------------|
        | 0      | 2                    |
        | 1      | 3                    |
        | 2      | 2                    |
        | 3      | 0                    |
        | 4      | 0                    |
        | 5      | 0                    |
        | 6      | 0                    |
        | 7      | 0                    |
        | 8      | 1                    |
        | 9      | 1                    |
        ---------------------------------
        >>> hist = [2, 3, 2, 0, 0, 0, 0, 0, 1, 1]
        >>> hist = hist.astype("float")
        ... hist /= (hist.sum() + 1e-7)  
        hist = hist = [0.222, 0.333, 0.222, 0, ..., 0.111, 0.111]   
    """
    return hist

#The extract_intensity_histogram function calculates Histogram for brightness (intensity) of the image (after converting it to grayscale).
"""
### Why use Intensity Histogram?
        - With your own hands an idea of the distribution of lighting in the photo.
        - Very useful in agricultural or medical images, because the disease sometimes changes the brightness of parts of the leaf.
        - Easy and fast in calculating.
        - It complements LBP, which focuses more on texture.
"""
def extract_intensity_histogram(image, bins=32):
    # Check and convert data type if necessary
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # We convert the image from BGR to gray to focus on exposure only.
    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    """
    [gray]: Gray image as source.
    [0]: Color channel used (brightness channel).
    None: No mask (meaning we count for each image).
    [bins]: The number of digits in which we divide values (by default 32 digits).
    [0, 256]: Range of brightness (from black = 0 to white = 255).
    flatten(): We convert it from a 2D array (z [32, 1]) to vector 1D (z [32]) so it stays easy to use as a feature.
    
    Example:
    >>> [gray] =[[  0,  50, 100, 150],
                [200, 255,  50, 100],
                [150, 200,  0, 255],
                [ 50, 100, 150, 200]]
    >>> hist = cv2.calcHist([gray], [0], None, [4], [0, 256])

    Here bins = 4 → its meaning is to divide brightness degrees into 4 groups:
        1- [0-64)
        2- [64-128)
        3- [128-192)
        4- [192-256)

        | Bin Range    | number of Pixels   |        Notes                    |
        |--------------|--------------------|---------------------------------|
        | 0-64         | 4                  | values: 0, 50, 50, 50           |
        | 64-128       | 3                  | values: 100, 100, 100           |
        | 128-192      | 3                  | values: 150, 150, 150           |
        | 192-256      | 6                  | values: 200, 200, 200, 255, 255 |
        -----------------------------------------------------------------------

    >>> hist = [4.0, 3.0, 3.0, 6.0]
    >>> cv2.normalize(hist, hist).flatten()
    ... hist = [0.2, 0.15, 0.15, 0.3] Sum = 1
    """
    return hist

def extract_features_from_folder(folder_path, image_size=(128, 128)):
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
        >>> classes = os.listdir(folder_path) 
        >>> classes = ["Pepper__bell___Bacterial_spot","Pepper__bell___healthy","PlantVillage", ... ...]

    >>> os.path.join ==> combine the folder path with inner folder to get their path
    Example:
        >>> folder_path = "PlantVillage"
        ... label = "Apple___Black_rot"
        ... full_path = os.path.join(folder_path, label)
        >>> full_path
        PlantVillage\Apple___Black_rot
    ------------------------------------------------------------------------------------------------------------

    """
    features = []
    labels = []
    
    classes = os.listdir(folder_path)

    for label_name in classes:

        label_folder = os.path.join(folder_path, label_name)
        if not os.path.isdir(label_folder):
            continue

        for img_name in os.listdir(label_folder):
            img_path = os.path.join(label_folder, img_name)
            img = cv2.imread(img_path) # read and convert into array using cv2 = openCV lib (250, 250, 3)

            if img is None:
                continue

            img = cv2.resize(img, image_size) # resize images into (128, 128)
            """
            You asked:
            "How is it even possible to resize a (250, 250, 3) image down to (128, 128, 3)?"
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
            >>> img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)
            """

            lbp_feat = extract_lbp_features(img)
            hist_feat = extract_intensity_histogram(img)

            feature_vector = np.hstack([lbp_feat, hist_feat]) # combine the two features
            features.append(feature_vector)
            labels.append(label_name)

    # Encode labels
    le = LabelEncoder()# convert the classes into int 
    encoded_labels = le.fit_transform(labels)
    
    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    features, encoded_labels = smote.fit_resample(features, encoded_labels)
    
    return np.array(features), encoded_labels, le

features, labels, label_encoder = extract_features_from_folder(r'../data/PlantVillage')