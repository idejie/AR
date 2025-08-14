# AR-VRM: Imitating Human Motions for Visual Robot Manipulation with Analogical Reasoning

[Arxiv Paper](https://arxiv.org/abs/2508.07626)
> By Dejie Yang, Zijing Zhao, Yang Liu $^*$
>
> Accepted By ICCV 2025

## 1. Abstract

Visual Robot Manipulation (VRM) aims to enable a robot to follow natural language instructions based on robot states and visual observations, and therefore requires costly multi-modal data. To compensate for the deficiency of robot data, existing approaches have employed vision-language pretraining with large-scale data. However, they neither utilize web data that differs from robotic tasks, nor train the model in an implicit way (e.g., predicting future frames at the pixel level), thus showing limited generalization ability under insufficient robot data. In this paper, we propose to learn from large-scale human action video datasets  in an explicit way (i.e., imitating human actions from hand keypoints), introducing Visual Robot Manipulation with Analogical Reasoning (AR-VRM). To acquire action knowledge explicitly from human action videos, we propose a keypoint Vision-Language Model (VLM) pretraining scheme, enabling the VLM to learn human action knowledge  and directly predict human hand keypoints. During fine-tuning on robot data, to facilitate the robotic arm in imitating the action patterns of human motions, we first retrieve human action videos that perform similar manipulation tasks  and have similar historical observations , and then learn the Analogical Reasoning (AR) map between human hand keypoints and robot components. Taking advantage of focusing on action keypoints instead of irrelevant visual cues, our method achieves leading performance on the CALVIN benchmark {and real-world experiments}. In few-shot scenarios, our AR-VRM outperforms previous methods by large margins , underscoring the effectiveness of explicitly imitating human actions under data scarcity.

![teaser](https://github.com/user-attachments/assets/92593306-6479-4d30-8f17-f8b0469929c8)


## 2. Benchmark

Download and unzip the [CALVIN](https://github.com/mees/calvin) dataset. 
```bash
$ git clone --recurse-submodules https://github.com/mees/calvin.git
$ export CALVIN_ROOT=$(pwd)/calvin
$ cd $CALVIN_ROOT/dataset
$ sh download_data.sh ABC | ABCD
$ python <Project_Root_Dir>/data/calvin2lmdb.py --input_dir <dir> 
```


## 3. Requirement
### 3.1 Base

- Python>=3.10
- CUDA>=12.2
- More Detials in `requirements.txt`
### 3.2 Python dependencies
```bash
$ git clone --recurse-submodules https://github.com/idejie/ar.git
$ export AR_ROOT=$(pwd)/ar
$ cd $AR_ROOT
$ pip install -r requirements.txt
```
### 3.3  Calvin dependencies

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

## 4. Train

```bash
# for setting ABC-D
 accelerate launch --config_file configs/multiple.yaml --main_process_port 0 train.py --config configs/configs_ABC.json # or configs_ABCD



```

## 5. Evaluation


```bash
# for setting ABC-D
 accelerate launch --config_file configs/multiple.yaml --main_process_port 0 evaluate.py --config configs/configs_ABC.json # or configs_ABCD

```


## 6. Results and Checkpoints

- path_config: reference to `configs`



| Task Setting|  Avg Len. |  Checkpoints     |
| -------- |  -------------- |-------------- |
| ABCD-D |     4.27     |     [ABC_ckpts]( https://pan.baidu.com/s/17El0b3xJh8cU5129-m9AMQ?pwd=v74f)/[GDrive-Comingsoon]()         | 
| ABC-D |     3.29     |     [ABC_ckpts]( https://pan.baidu.com/s/17El0b3xJh8cU5129-m9AMQ?pwd=v74f) /[GDrive-Comingsoon]()           | 


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
