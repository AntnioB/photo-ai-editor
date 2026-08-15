#imports
import oss

#variables
rawList = []
editedList= []

#File Matching & Indexing: Scans your raw and edited folders, verifies both images exist for every pair, and builds an indexed master list of matching file paths based on filenames.
with os.scandir('../data/raw') as d:
    for e in d:
        rawList.append(e.name)
    
with os.scandir('../data/edited') as d:
    for e in d:
        editedList.append(e.name)

if rawList.count() == editedList.count():
    raise Exception("Raw number of images does not match edited number of images")



#Image Loading: Reads the unedited JPEG and the corresponding edited target image into memory using Python libraries like Pillow or OpenCV.

#Spatial Resizing & Alignment: Downsamples both images to a uniform resolution (such as $256 \times 256$) so they can be grouped into fixed-size training batches, ensuring identical aspect ratios via consistent cropping or letterboxing.

#Tensor Conversion & Normalization: Converts image arrays into PyTorch Tensors and scales RGB pixel values from standard $0\text{--}255$ integers into a normalized $0.0\text{--}1.0$ floating-point range.

#Synchronized Data Augmentation: Applies identical random transformations (such as horizontal flips or minor rotations) to both the unedited image and the target edit simultaneously during training to artificially expand your dataset size.

#PyTorch Dataset Hooks: Provides the required internal methods—one to report the total count of paired images, and another to retrieve a single preprocessed (unedited_tensor, edited_tensor, metadata) item by index.