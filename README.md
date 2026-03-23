# SLAM With Movable Objects In Interactive Environment
[CH](README_CH.md) | [Paper](https://hdl.handle.net/11296/yqv8hh)


![Demo GIF](show/MO_SLAM(object-move).gif)

## Installation

```shell
git clone https://github.com/611223001/Movable-Objects-SLAM.git
```

1. Create a conda environment
```shell
conda create -n mo_slam python=3.10 -y
conda activate mo_slam
```

2. Install [detectron2](https://github.com/facebookresearch/detectron2)
```shell
pip install torch torchvision torchaudio

git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2

```

3. Install remaining dependencies
```shell
cd MovableObjectSLAM

# MovableObjectSLAM$
conda env update -n mo_slam -f environment.yml --prune
```

### Test

```shell
# MovableObjectSLAM$
python main.py
```

## Introduction

This project implements a monocular SLAM system for interactive indoor environments. The goal is to maintain object consistency in the map even when objects may be moved and later become static again.

Unlike traditional SLAM methods that treat dynamic objects as interference and directly discard them, this system incorporates movable objects into the map representation. Through object-level modeling and state management, it preserves object identity before and after movement. The system can temporarily isolate unstable observations while an object is moving, and re-estimate the object pose after it becomes static again so that it can be reintegrated into the map.

### Features

- Supports object-level map representation and identity consistency maintenance
- Handles object state transitions from static, to moving, and back to static
- Combines geometric consistency for object pose re-estimation
- Built on top of a feature-based monocular SLAM framework
- Suitable for interactive indoor scenes containing movable objects

### Core Concept

Instead of treating a moved object as a completely new object, this system maintains map consistency through the following process:

1. Detect whether an object has moved
2. Temporarily remove unreliable dynamic observations
3. Re-estimate object pose after the object becomes static again
4. Reintegrate the object and its related landmarks back into the map

With this process, the system can maintain a reusable and consistent map representation in interactive environments.

### Limitations

Although this method attempts to incorporate movable objects into the map, it is still fundamentally based on feature-based monocular SLAM, and this type of framework is not ideal for robust object-level representation.

Because feature-based monocular SLAM builds sparse map points that only reflect local image features, it is difficult to effectively represent full object structure. As a result, additional object models are required as supplements. However, this design is not fully compatible with the original landmark-centered map representation, and it also makes loop closing difficult to integrate naturally.

In addition, loop closing is currently not integrated, so long-term operation may produce accumulated drift. Overall performance also depends on segmentation quality and object observation conditions. When segmentation is unstable or observations are insufficient, results are affected.

Therefore, this method is better viewed as a feasibility exploration rather than a mature and robust object SLAM solution.

### Future Directions

To improve stability and extensibility, a more reasonable direction is to further redesign the map representation and perception strategy.

One possible direction is to use foundation vision models. Compared with approaches that rely on fixed categories and segmentation outputs, foundation vision models may provide more stable object features and stronger cross-view correspondence, so object tracking, identity maintenance, and state estimation do not depend as heavily on fragile segmentation and local geometric conditions.

Another direction is to use Gaussian Splatting SLAM as the foundation. Compared with feature-based SLAM that only builds sparse map points, Gaussian Splatting methods can provide representations closer to object surfaces and appearance, which is theoretically more suitable for direct object-level modeling. If movable-object consistency mechanisms are built on top of this representation in the future, the system should become more natural in object tracking, reconstruction, and map consistency than the current feature-point-based plus extra object-model architecture.
