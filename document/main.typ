#import "template/lib.typ": *

#show: project.with(
  title: "Optical Flow FlowNet",
  author: "Alejandro D.J Gomez Florez",
  date: "November 26, 2025",
  chair: "Ms Applied Mathematics",
)

= Introduction

This document presents the development of a model to estimate vehicle velocity using a single camera. The model is based on the Optical Flow FlowNet architecture.

Optical flow is employed because, from the perspective of a fixed camera, the motion of objects in a scene can be represented as a flow field. Optical-flow methods estimate object motion from sequences of images by tracking consistent changes in pixel intensity over time. As an illustration, @fig:optical_flow_example shows an example of a typical visual representation of optical flow, taken from the article [Optical Flow with RAFT](https://medium.com/data-science/optical-flow-with-raft-part-1-f984b4a33993).

#figure(
  image("pictures/optical_flow_example.png", width: 80%),
  caption: [Optical Flow Example],
) <fig:optical_flow_example>

One of the main problems in using optical flow is the computational cost of implementing it in real-time applications. Classical methods such as Lucas–Kanade or Horn–Schunck are widely used in this field and typically operate on the entire image, which leads to a high computational cost. Although these methods can estimate the flow field, it is still necessary to relate this information to a vehicle in motion, with the velocity of the vehicle, because the camera is mounted on it. Therefore, a method is required to infer the vehicle's motion and estimate its velocity from the optical-flow information captured by the onboard camera.

For this development, a first approach was made using a first notebook calculating the optical flow between two images of a vehicle in motion. A dataset from the [KITTI](https://www.cvlibs.net/datasets/kitti/) was used to train and test the classical algorithm. The notebook is available in the [OpticalFlowFlowNet](https://github.com/aldajo92/OpticalFlowFlowNet) repository.