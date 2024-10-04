# PyYel
*PyYel* is a personnal library that aims at helping the deployement of strong data science tools, from data handling to deep learning.

## Quick start
1. Install the library.

``` bash
your_path> pip install PyYel
```

2. Import the library into you code.

``` python
import pyl
```

3. Import the relevant features.

``` python
from pyl.models.LLM import LLMDecodingPhi, LLMEncodingBARTLargeMNLI
from pyl.models.CNN import CNNClassificationResNet
```

## Content

### Data
A collection of features to manipulate the data. Can be used to implement pipelines, preprocessing, data augmentation...

- **Augmentations:** a compilation of classes featuring methods to augment a datapoint of various type.
    - ImageAugmentation : features a handfull of functions that can augment any type of data, as well as its labels.
    - TODO

- **Reduction:** acompilation of classes featuring methods to reduce datapoint of various type.
    - TODO/TO-REWORK

- **Utils:** a collection of powerful tools that permit an easy manipulation of the datapoints.
    - TODO/TO-REWORK

### Models
The neural networks implementations. These are grouped by types and tasks.

- **CNN (Convolutional Neural Networks)**

|Source model|PyYel model|Task|Status|
|------------|-----------|----|------|
|ResNet|CNNCLassificationResNet|Classification|Implemented|
|FasterRCNN|CNNDetectionFasterRCNN|Detection|Implemented|
|SSD|CNNDetectionSSD|Detection|Implemented|
|RetinaNet|CNNDetectionRetinaNet|Detection|TODO|
|/|CNNKeypoint|Keypoint detection|TODO|
|FCN|CNNSegmentationFCN|Segmentation|Implemented/TODO|
|DeeplabV3|CNNSegmentationDeeplabV3|Segmentation|Implemented/TODO|

**Note:** _Traditionnal computer vision networks. Features a model builder to design custom small-sized networks._

- **FCN (Fully Connected Networks)**

|Source model|PyYel model|Task|Status|
|------------|-----------|----|------|
|/|FCNBuilder|/|TODO|
**Note:** _Dense models. Features a model builder to design custom small-sized networks._

- **LLM (Large Language Models)**

|Source model|PyYel model|Task|Status|
|------------|-----------|----|------|
|Mistral7B v0.1|LLMDecodingMistral7B|Decoding: text-to-text generation|Implemented|
|OPT 125M|LLMDecodingOPT125m|Decoding: text-to-text generation|Implemented|
|Phi 3.5 Mini Instruct|LLMDecodingPhi|Decoding: text-to-text generation|Implemented|
|Phi 3.5 MoE|LLMDecodingPhiMoE|Decoding: text-to-text generation|Implemented/TODO|
|BART Large|LLMEncodingBARTLargeMNLI|Encoding: zero-shoot classification|Implemented|
|DeBERTaV3 Base|LLMEncodingDeBERTaV3Base|Encoding: zero-shoot classification|Implemented|
|DeBERTaV3 Base|LLMEncodingDeBERTaV3BaseMNLI|Encoding: zero-shoot classification|Implemented|
|DeBERTaV3 Large|LLMEncodingDeBERTaV3Large|Encoding: zero-shoot classification|Implemented|

**Note:** _NLP transformers._

- **LVM (Large Vision Models)**

|Source model|PyYel model|Task|Status|
|------------|-----------|----|------|
|ViT|LVMVisionTransformerClassification|Classification|TODO|

**Note:** _Computer vision transformers._


### Utils
A collection of higher-level tools, that simplifies the manipulation of the library 

#### TODO/TO-REWORK

## Notes
TODO