# AR-VRM: Imitating Human Motions for Visual Robot Manipulation with Analogical Reasoning
> By Dejie Yang, Zijing Zhao, Yang Liu $^*$
>
> Accepted By ICCV 2025

## 1. Abstract

Visual Robot Manipulation (VRM) aims to enable a robot to follow natural language instructions based on robot states and visual observations, and therefore requires costly multi-modal data. To compensate for the deficiency of robot data, existing approaches have employed vision-language pretraining with large-scale data. However, they neither utilize web data that differs from robotic tasks, nor train the model in an implicit way (e.g., predicting future frames at the pixel level), thus showing limited generalization ability under insufficient robot data. In this paper, we propose to learn from large-scale human action video datasets  in an explicit way (i.e., imitating human actions from hand keypoints), introducing Visual Robot Manipulation with Analogical Reasoning (AR-VRM). To acquire action knowledge explicitly from human action videos, we propose a keypoint Vision-Language Model (VLM) pretraining scheme, enabling the VLM to learn human action knowledge  and directly predict human hand keypoints. During fine-tuning on robot data, to facilitate the robotic arm in imitating the action patterns of human motions, we first retrieve human action videos that perform similar manipulation tasks  and have similar historical observations , and then learn the Analogical Reasoning (AR) map between human hand keypoints and robot components. Taking advantage of focusing on action keypoints instead of irrelevant visual cues, our method achieves leading performance on the CALVIN benchmark {and real-world experiments}. In few-shot scenarios, our AR-VRM outperforms previous methods by large margins , underscoring the effectiveness of explicitly imitating human actions under data scarcity.

## 2. Benchmark

Download and unzip the [CALVIN](https://github.com/mees/calvin) dataset. 
```bash
$ git clone --recurse-submodules https://github.com/mees/calvin.git
$ export CALVIN_ROOT=$(pwd)/calvin
$ cd $CALVIN_ROOT/dataset
$ sh download_data.sh D | ABC | ABCD 
```

## 3. Requirement

- Python>=3.10
- CUDA>=12.2
- More Detials in `requirements.txt`

```bash
$ git clone --recurse-submodules https://github.com/idejie/AR-VLA
$ export AR_ROOT=$(pwd)/AR-VLA
$ cd $AR_ROOT
$ pip install -r requirements.txt
```

## 4. Train

```bash
# for setting ABCD-D with DP
bash ./scripts/train.sh  ./configs/ABCD_DP.json
# for setting ABC-D with DP
bash ./scripts/train.sh  ./configs/ABC_DP.json

```

## 5. Evaluation


### 5.1 dependencies

```bash
$ git clone --recurse-submodules https://github.com/mees/calvin.git
$ export CALVIN_ROOT=$(pwd)/calvin
$ cd $CALVIN_ROOT
$ onda create -n calvin_venv python=3.8  # or use virtualenv
$ conda activate calvin_venv
$ pip install setuptools==57.5.0
$ sudo apt-get -y install libegl1-mesa libegl1 libgl1 libosmesa6-dev  ffmpeg  patchelf
$ sh install.sh

```
### 5.2 scripts
```bash
# for setting ABCD-D with DP
bash ./scripts/eval.sh  ./configs/ABCD_DP.json
# for setting ABC-D with DP
bash ./scripts/eval.sh  ./configs/ABC_DP.json

```


## 6. Results and Checkpoints


| Task Setting| Policy | Avg Len. |Avg Len. |  Checkpoints     |
| -------- | -------------- | -------------- | -------------- |-------------- |
| ABCD-D | DP|      4.34      |  86.75% |    [ABCD_DP]()        | 
| ABC-D | DP|     3.29     |     65.80%  |   [ABC_DP]()         | 

## 7. Citation
If you find our paper or project useful, please cite our work by the following BibTeX:

```
@inproceedings{yang2025ar,
  title={AR-VRM: Imitating Human Motions for Visual Robot Manipulation with Analogical Reasoning},
  author={Dejie Yang,  Zijing  Zhao, Yang Liu},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2025}
}
```