# Keras RetinaNet [![Build Status](https://travis-ci.org/fizyr/keras-retinanet.svg?branch=master)](https://travis-ci.org/fizyr/keras-retinanet) [![DOI](https://zenodo.org/badge/100249425.svg)](https://zenodo.org/badge/latestdoi/100249425)

Keras implementation of RetinaNet object detection as described in [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
by Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He and Piotr Dollár.

## Installation

1) Clone this repository.
2) Ensure numpy is installed using `pip install numpy --user`
3) In the repository, execute `pip install . --user`.
   Note that due to inconsistencies with how `tensorflow` should be installed,
   this package does not define a dependency on `tensorflow` as it will try to install that (which at least on Arch Linux results in an incorrect installation).
   Please make sure `tensorflow` is installed as per your systems requirements.
4) Alternatively, you can run the code directly from the cloned  repository, however you need to run `python setup.py build_ext --inplace` to compile Cython code first.
5) Optionally, install `pycocotools` if you want to train / test on the MS COCO dataset by running `pip install --user git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI`.

## Testing
An example of testing the network can be seen in [this Notebook](https://github.com/delftrobotics/keras-retinanet/blob/master/examples/ResNet50RetinaNet.ipynb).
In general, inference of the network works as follows:
```python
boxes, scores, labels = model.predict_on_batch(inputs)
```

Where `boxes` are shaped `(None, None, 4)` (for `(x1, y1, x2, y2)`), scores is shaped `(None, None)` (classification score) and labels is shaped `(None, None)` (label corresponding to the score). In all three outputs, the first dimension represents the shape and the second dimension indexes the list of detections.

Loading models can be done in the following manner:
```python
from keras_retinanet.models import load_model
model = load_model('/path/to/model.h5', backbone_name='resnet50')
```

Execution time on NVIDIA Pascal Titan X is roughly 75msec for an image of shape `1000x800x3`.

### Converting a training model to inference model
The training procedure of `keras-retinanet` works with *training models*. These are stripped down versions compared to the *inference model* and only contains the layers necessary for training (regression and classification values). If you wish to do inference on a model (perform object detection on an image), you need to convert the trained model to an inference model. This is done as follows:

```shell
# Running directly from the repository:
keras_retinanet/bin/convert_model.py /path/to/training/model.h5 /path/to/save/inference/model.h5

# Using the installed script:
retinanet-convert-model /path/to/training/model.h5 /path/to/save/inference/model.h5
```

Most scripts (like `retinanet-evaluate`) also support converting on the fly, using the `--convert-model` argument.


## Training
`keras-retinanet` can be trained using [this](https://github.com/fizyr/keras-retinanet/blob/master/keras_retinanet/bin/train.py) script.
Note that the train script uses relative imports since it is inside the `keras_retinanet` package.
If you want to adjust the script for your own use outside of this repository,
you will need to switch it to use absolute imports.

If you installed `keras-retinanet` correctly, the train script will be installed as `retinanet-train`.
However, if you make local modifications to the `keras-retinanet` repository, you should run the script directly from the repository.
That will ensure that your local changes will be used by the train script.

The default backbone is `resnet50`. You can change this using the `--backbone=xxx` argument in the running script.
`xxx` can be one of the backbones in resnet models (`resnet50`, `resnet101`, `resnet152`), mobilenet models (`mobilenet128_1.0`, `mobilenet128_0.75`, `mobilenet160_1.0`, etc), densenet models or vgg models. The different options are defined by each model in their corresponding python scripts (`resnet.py`, `mobilenet.py`, etc).

Trained models can't be used directly for inference. To convert a trained model to an inference model, check [here](https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model).


Intelligent Automation for Structured Data Extraction from PDF documents.


Abstract
In most industries and organizations, a large part of organizational knowledge resides in documents, but these documents are in unstructured form. Extracting data from such documents has been difficult. One of those industries is the life sciences, and one of the major processes in life sciences is to bring a drug to market with safety and efficacy. 
The unique nature of this domain is that on the one hand, we want to ensure safety because it can adversely affect human lives, and on the other, extract maximum value out of the drugs, experiments, and clinical trials.
Traditional systems to support the clinical trials collect data in a document-centric way. Much of this information ends up as PDF documents that are scanned and used in the business process.
The information from these PDF documents is then manually entered into systems for analytics and further processing, which is an expensive exercise.  AI, with its advances in robust optical character recognition and natural language processing, can make this conversion much more efficient and accurate.  However, extracting data from clinical forms that includes labels, checkboxes, tables, and lines are more challenging because we need to extract the labels and the corresponding values.
At DataFoundry, we have created an intelligent data extraction system using AI technologies to automate the data extraction to any one of many structured formats.  The system does minimal manual annotations to capture the semantics of specific sections for any particular document template.  Once that has been done then millions of documents can be fed through the system to extract information automatically.
In this paper, we will describe the business drivers behind such a system, the architecture of the system, show how the system performs compared to human levels, and also showcase examples of documents processed. We will discuss the difficulties around exacting information from checkboxes and share details about the neural network architecture we used to achieve high accuracy.

Background
One of the challenges in the field of Document Analysis is, document structure detection. Here, techniques from computer vision, a sub-discipline of artificial intelligence, are used for (1) information detection and (2) extraction. In their article “Histograms of oriented gradients for human detection” [1]  Navneet Dalal and Bill Triggs describe the histogram of oriented method to detect the objects from images. While this approach addresses many practical object detection and extraction problems, more advanced techniques use deep neural networks, yet another sub-discipline of AI, to address more complex problems. Such classical deep convolutional neural networks extract feature maps to extract more complex objects as shown in the work by Pierre Sermanet, David Eigen, Xiang Zhang, Michael Mathieu, Rob Fergus and Yann LeCun [2].  Specifically for document analysis, Neural Networks such as RCNN [3], fast RCNN [4], Faster RCNN [5], Yolo [6], RetinaNet [7,8] and then CornerNet [9] has shown good results to detect and extract complex document content such as graphs, figures, tables, checkboxes etc. 
Checkbox detection is a unique problem in document analysis, because in addition detecting the checked box, we also need to detect the text which corresponds to the checked box. In his Master’s thesis “Optical character recognition for checkbox detection” [10] the author uses a three-step procedure for checkbox detection: 
1. image preprocessing, 
2. box detection through optical character recognition (OCR), and,
3. checkmark detection through optical mark recognition. 

IBM Knowledge Center [11] also proposes OCR usage for box detection, but they use a pixel threshold evaluation method, wherein the ratio of black to white pixels of the box area determines whether the box is checked. Another approach was offered by the authors of “Automatic Recognition Method for Checkbox in Data Form Image” [12]. They detect skews in image and then use the handwritten symbol recognition to solve the problem. However, this works only for handwritten boxes checked. 
However, none of the approaches solve the problem of the text detection in the checked box. In the clinical trial life sciences space specifically, our analysis of different documents shows, that text can be both on the left and on the right of the checkbox. The RCNN (recurrent convolution neural network) family can be implemented for this task, but since they are based on ruled base Selective Search or Edge Boxes detection algorithms, they are extremely time and resources consuming. After analyzis and experimentation to find a more efficient approach, we implemented a variation of the RetinaNet NN model which achieved high performance mean Average Precision (mAP). 

Problem Statement
The intent of the document analysis project was to create a web application, which will disaggregate multiple PDF format documents of a same type and reaggregate the informational content to make it identifiable, retrievable and easily readable. Generally, data in the pdf documents is semi-structured. As opposed to well-structured data, which conforms to a schema or data model and can be queried using structured query language to answer questions, the content does not adhere to any rigorous format.  

Solution 
To extract structured data as described in problem statement, we used the following steps:
Step 1:
Annotate a single document of each type
Step 2:
Extract data from multiple documents in accordance with initial annotation	Step 3:
Identify only information from checked boxes for extraction	Step 4:
Create structured data in a csv format 

Step 1 of the algorithm is to annotate a single document of a given type. See Figure 1 for examples of uploaded and annotated documents.


                                




Figure 1: An example of an unannotated document (left)  and annotated document (right).


The document are annotated so that the key and value pairs would be identifiable. For example, (name, Smith), and (age, 58 years) where the key is the question, and value is the corresponding answer. In Figure 1 we show the annotated keys with red boxes, and the annotated values with blue boxes. 

Step 2. Using the Python programming library fitz, we can detect the red and blue boxes and extract their corresponding coordinates. We also detect the text within the boxes and extract them into a Python data structure. Thus, keeping the coordinates of the boxes, we can extract the information from the boxes from any document of the same type. As you can see in Figure 2, the data from three analogous documents for three different patients was extracted from the annotated boxes into a single document.
 

Figure 2: Structured data received from three documents of the same type.

However, to retrieve only information from a checked box is yet another challenge. In Figure 3 we can see that for section “Sex” both options – male and female are retrieved. Information from the annotated boxes is retrieved fully irrespective of the checked box in front of only one of the options. That is a problem.

Step 3. To overcome that problem and detect the box that is checked, we used a RetinaNet deep learning network. We propose a completely new solution to the problem of checkbox detection: instead of detecting the box itself, we trained the network in a manner, that it could identify the bounding box of the text near the checked box (see Figure 3). This facilitates the work, because it reduces the number of steps in the code. It also gives a possibly higher accuracy, since we don’t have to rely on engineering solutions to find out whether the check box is on the left of the text or on the right of it.

 	 
Figure 3: Detection of the text near the checked box using RetinaNet deep learning network.

We have also considered cases where there is an additional information not only near the checkboxes, or the words selected are circled instead. For instance, in Figure 4, “56 years” is the output of our program. 
 
Figure 4: Detection of circled words and additional words.
After coordinate detection of the bounding box of the text, we proceed to the Step 4. The image is cropped and is passed to pytesseract Python library, which uses tesseract OCR. Tesseract is an  optical character recognition engine released under the Apache License and its development has been sponsored by Google since 2006. The OCR transferred text is then inserted to the corresponding space in the data structure shown in Figure 3. 

Data Preprocessing for RetinaNet Neural Network
The data for the network was collected from various documents containing different type of checkboxes (Figure 3 and Figure 4). Checkmarks inside the boxes also vary, being X-shaped or V-shaped. Some noise, like dots of different colors, were added to the images to avoid overfitting. Negative examples included both – images with all unchecked boxes and images without any checkboxes on them.

For positive data, we wrote code using pytesseract python library to give as an output an hOCR file. hOCR is an open standard of data representation for formatted text obtained from optical character recognition, which among many other things contains the bounding box coordinates of the words [14]. For each part of the image, the bounding box coordinates of the words were found and were given as labels of the data. Three different type of classes were created: one for words near the checkboxes for cases represented in Figure 3, and two additional classes for cases presented in Figure 4: one for the text input without checkboxes, and the other for circled words. We named the classes “yes,” “ok,” and “ellipse.” You can see an example of preprocessed data in Figure 5.
Checkbox image	Xmin	Ymin	Xmax	Ymax	Class name
 	505	18	670	55	tick
					
 	80	25	135	65	notation
	660	10	77	65	circle
					
 	330	15	480	50	tick
					
 	1340	15	1580	50	tick
					
 	1385	5	1620	95	tick
					
 	80	25	135	65	notation
	660	10	770	65	circle
					
 	700	10	890	95	tick
					
 	85	25	140	65	notation
	660	10	770	65	circle
					
 	70	25	125	65	notation
	660	10	770	65	circle
					
 	840	25	1190	70	tick
					
 	1100	20	1225	55	tick
Figure 5: Preprocessed data for RetinaNet deep learning neural network.

Each of the data lines in Figure 5 contains the x,y coordinates of upper left and lower right corners of the bounding boxes of the label data (colums 2 through 5), and their respective classes (column 6). The data was checked for avoiding possible mistakes and making it work for RetinaNet  through keras-retinanet/keras_retinanet/bin/debug.py script designed for debugging images by RetinaNet model developers. To make text detectable by the neural network, anchor parameters of anchor boxes were set to ratios = np.array([0.01,0.03,0.09,0.27, 1], keras.backend.floatx()), scales = np.array([20,  2(1.0 / 3.0),  2(2.0 / 3.0),  3.5] in keras-retinanet/keras_retinanet/utils/anchors.py python script. All the coordinates were verified through debug.py

RetinaNet Training
One of peculiarities of RetinaNet model is the usage of   Focal Loss (FL(pt) = −(1 – pt) γ *log(pt)) function during the classification [15]. This conceptual interpretation of this function is that it increases the overall contribution of positive examples in the training process. Data imbalances of having either too many positively tagged data or too many negatively tagged data can result in machine learning models trying to generalize too much, which is why they have to be ‘balanced’ first. The Focal Loss is designed to address the one-stage object detection scenario in which there is an extreme imbalance between foreground and background classes during training (e.g., 1:1000). 

Training data contained 1205 samples. The parameters of training are as follows: batch_size=1, steps=1249,  epochs=50. We used pretrained COCO weights for RetinaNet.

We used random-transform and no-resize parameters while training. Random transform usage avoids the need to specify the min_rotation, max_rotation, min_translation, max_translation, min_shear, max_shear, min_scaling, max_scaling, flip_x_chance,  flip_y_chance transformation parameters on initial pictures. No-resize is designed to feed the network with pictures of original sizes, and not set one-size-for-all to them

Evaluation metrics
The mean average precision (mAP) is a main evaluation metric for RetinaNet [15]. The tables of Precision and Recall are calculated for mAP estimation.  We assumed that score_threshold=0.4 and iou_threshold=0.5. 
Rank	Precision	Recall	True_positives
1.		1.00	0.00	1.0
2.		1.00	0.01	2.0
3.		1.00	0.01	3.0
…	…	…	…
150.		1.00	0.56	150.0
151.		1.00	0.57	151.0
152.		1.00	0.57	152.0
…	…	…	…
282.		0.90	0.95	254.0
283.		0.90	0.95	254.0
284.		0.89	0.95	254.0
For class “tick"   average_precision 0.9483, num_annotations 267.0
Figure 6: Evaluation metrics for class “tick”

Rank	Precision	Recall	true_positives
1.		1.00	0.01	1.0
2.		1.00	0.02	2.0
3.		1.00	0.03	3.0
…	…	…	…
50.		1.00	0.51	50.0
51.		1.00	0.52	51.0
52.		1.00	0.53	52.0
…	…	…	…
96.		0.98	0.96	94.0
97.		0.98	0.97	95.0
98.		0.98	0.98	96.0
For class “notation"   average_precision 0.9753, num_annotations 98.0

Figure 7: evaluation metrics for class “notation”

Rank	Precision	Recall	true_positives
1.		1.00	0.02	1.0
2.		1.00	0.04	2.0
3.		1.00	0.06	3.0
…	…	…	…
22.		1.00	0.47	22.0
23.		1.00	0.49	23.0
24.		1.00	0.51	24.0
…	…	…	…
45.		1.00	0.96	45.0
46.		1.00	0.98	46.0
47.		1.00	1.00	47.0
For class “circle"   average precision  1, num_annotations 98.0, num_annotations 47.0

Figure 8: Evaluation metrics for class “circle”
Based on the tables we found mAP=(0.9483+ 0.9753+ 1.0000)/3=0.9745 for test set and analogically for train set we have mAP=0.9816 which are very good.

Conclusion
Automating document processing in life sciences for processing clinical trials data is quite challenging. The semi structured nature of PDF documents make it difficult to have a rule-based approach to extract name value pairs, that can then in turn be fed into a structured database for further processing.  Machine learning and specifically deep learning approaches hold promise to provide better results.

Checkbook box detection is among the most challenging because the text corresponding to the check box may be on the right, or on the left, or any other location close to the check box itself. In addition, the variation of how the box is checked is high. There is no way to programmatically identify the box’s label and its value.

We used a modified RetinaNet model for detecting/predicting checkboxes and their corresponding values. RetinaNet outputs the probabilities of checkboxes (i.e. the probability of existence of a checkbox), and we consider boxes as checked if the probabilities are higher than 0.5. It has been shown that our model detects checkboxes with high accuracy, approximately 90% on a given test data.

While this experiment was done on domain specific PDF documents, the same concept can be applied to documents in any other domain as long as they are trained using the appropriate deep neural network. Often in business, there is a lot of unstructured and semi structured documents (such as in Microsoft Word), and if we can systematically and automatically extract and tag that information, it will be valuable for the business leading to better and faster decision making, reduced cost, and reduced errors.













References
1.	Navneet Dalal and Bill Triggs Histograms of oriented gradients for human detection //2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05).

2.	Pierre Sermanet, David Eigen, Xiang Zhang, Michael Mathieu, Rob Fergus, Yann LeCun OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks // arXiv:1312.6229 [cs.CV] 21 Dec2013 

3.	Ross Girshick Jeff Donahue Trevor Darrell Jitendra Malik Rich feature hierarchies for accurate object detection and semantic segmentation // arXiv:1311.2524v5 [cs.CV] 22 Oct 2014

4.	Ross Girshick Fast R-CNN // arXiv:1504.08083v2 [cs.CV] 27 Sep 2015

5.	Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks // arXiv:1506.01497v3 [cs.CV] 6 Jan 2016

6.	Joseph Redmon, Santosh Divvala, Ross Girshick , Ali Farhadi You Only Look Once: Unified, Real-Time Object Detection // arXiv:1506.02640v5 [cs.CV] 9 May 2016

7.	Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollar, Focal Loss for Dense Object Detection // arXiv:1708.02002v2 [cs.CV] 7 Feb 2018

8.	Yixing Li, Fengbo Ren Light-Weight RetinaNet for Object Detection // arXiv:1905.10011v1 [cs.CV] 24 May 2019

9.	Hei Law • Jia Deng CornerNet: Detecting Objects as Paired Keypoints // arXiv:1808.01244v2 [cs.CV] 18 Mar 2019

10.	Istle, J. M. (2004). Optical character recognition for checkbox detection [Master's thesis]. https://digitalscholarship.unlv.edu/cgi/viewcontent.cgi?article=2723&context=rtds 

11.	IBM Knowledge Center https://www.ibm.com/support/knowledgecenter/en/SSZRWV_9.0.0/com.ibm.dc.develop.doc/dcadg363.htm 

12.	Zhang, S., Yuan, S., & Niu, L. (2014). Automatic Recognition Method for Checkbox in Data Form Image. 2014 Sixth International Conference on Measuring Technology and Mechatronics Automation. https://ieeexplore.ieee.org/abstract/document/6802658/authors#authors
13.	 Tesseract (OCR), Wikipedia, https://en.wikipedia.org/wiki/Tesseract_(software)
 
14.	 hOCR, Wikipedia, https://en.wikipedia.org/wiki/HOCR 

15.	Lin, T., Goyal, P., Girshick, R., He, K., & Dollar, P. (2017). Focal loss for dense object detection. 2017 IEEE International Conference on Computer Vision (ICCV). https://doi.org/10.1109/iccv.2017.324

16.	Mean average precision. (n.d.). SpringerReference. https://doi.org/10.1007/springerreference_65277

