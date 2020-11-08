# Project Write-Up

## Model Selection and Custom Layers
The model that was chosen is [faster_rcnn_inception_v2_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz) in [Tensorflow Object Detection Model Zoo.](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) This model gave better results compared with other model that I've tried, such as [ssd_mobilenet_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz) and [ssd_inception_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz).

Custom layers are a necessary and important to have feature of the OpenVINO™ Toolkit, although one shouldn’t have to use it very often, if at all, due to all of the supported layers. The list of [supported layers](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html) relates to whether a given layer is a custom layer. Any layer not in that list is automatically classified as a custom layer by the Model Optimizer.

For Tensorflow model, there are three options to converting custom layers:
1. [Register the custom layers as extensions to the Model Optimizer](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_customize_model_optimizer_Extending_Model_Optimizer_with_New_Primitives.html).
2. [Replace the unsupported subgraph with a different subgraph](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_customize_model_optimizer_Subgraph_Replacement_Model_Optimizer.html).
3. [Offload the computation of the subgraph back to TensorFlow during inference](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_customize_model_optimizer_Offloading_Sub_Graph_Inference.html).

OpenVINO™ contains extensions for custom layers of models in [Tensorflow Object Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md), including the model used in this project. These extensions configurations are located in `<OpenVINO install dir>/deployment_tools/model_optimizer/extensions/front/tf`.

To convert the model to Intermediate Representations (IR):
1. Download model  
```sh
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz`
```
2. Unzip the downloaded model  
```
tar -xvf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
```
3. In the directory of the extracted folder, run:  
```sh
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json`
```

The model are converted to IR in `bin` and `xml` files.

## Run the application

After starting the Mosca server, GUI and FFmpeg Server, open a new terminal and run the following command:  
```sh
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m model/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.45 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```

## Comparing Model Performance (With and without the Use of the OpenVINO™ Toolkit)

**Accuracy**
* Both models can detect all the six people occur in the video.

**Size**
* The file size of the model pre and post-conversion was almost the same. The faster_rcnn_inception_v2_coco model `pb` file is 54.5 MB and the IR `bin` file is size is 50.7 MB.

**Inference Time**  
* The average inference time of post-conversion model running on CPU of classroom workspace is 890ms. The performance can be improved by using other hardware such as CPU with external or integrated GPU, VPU and FPGA.

## Assess Model Use Cases

This type of application could be very useful in scenarios such as:
*  Retail Shops - to count the number of customers in the shop, the duration of their stay and which shelves they spent most time at.
* Bus Stops - to count the number of people at the bus stops, so that the control can dispatch more buses to meet the demand.
* Public places where crowd control is important.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. The potential effects of each of these are as follows:
* Lighting - Some preprocessing need to be done as the model may not identify a person if the lighting is too dim or too bright.
* Image size - Lower resolution of input image or video may affect the accuracy of the model.

## Reference
[Real-time Human Detection in Computer Vision — Part 2](https://medium.com/@madhawavidanapathirana/real-time-human-detection-in-computer-vision-part-2-c7eda27115c6)
