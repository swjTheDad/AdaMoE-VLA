# Expertise Need Not Monopolize: Action-Specialized Mixture of Experts for Vision-Language-Action Learning😋

[\[📖 Arxiv\]](https://arxiv.org/abs/2510.14300)
[\[🌐 Website\]](https://charleshen1412.github.io/AdaMoE-VLA/)
## ✅ To Dos
- ✅ Paper Release
- ✅ Website Release
- ✅ Release Training and Evaluation Code
- [ ] Release Experts Activation Visualization
- [ ] Release Checkpoints

## 🎬 Introduction
This is the official repository for the paper [Expertise Need not Monopolize: Action-Specialized Mixture of Experts for Vision-Language-Action Learning](https://arxiv.org/abs/2510.14300), in which we explore efficient ways to scale up Vision-Language-Action(VLA) Models via Mixture-of-Experts(MoE) architectures. We build our code based on the official [openpi repo](https://github.com/Physical-Intelligence/openpi), and replaced the feedforward network of the openpi model's action expert with sparse activated MoE. 
Our key finding is that the original coupled design of routers in MoE limits model performance. Therefore, we propose a simple yet effective modification that decouples expert selection from expert weighting through the introduction of a scale adapter. We call our new architecture **AdaMoE-VLA**.
![Pipeline](img/pipeline.png)
Beyond resolving the optimization conflict, this design embodies our core philosophy: **“Expertise need not monopolize”**—the ability of an expert to be selected for a task should not dictate its relative importance in the final output. An expert might be highly relevant (selected by the router) while still contributing modestly (controlled by both the scale adapter and router), or vice versa. This decoupling allows for more nuanced expert combinations that better reflect the complex, multi-faceted nature of robotic manipulation tasks.
More details and visualizations can be found in [our paper](https://arxiv.org/abs/2510.14300) and [our website](https://charleshen1412.github.io/AdaMoE-VLA/).

## 🔥 Quick Start

### 🌏 Environment Setup
Before running uv commands, please make sure [uv](https://docs.astral.sh/uv/) was installed in your machine. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up.
```
git clone --recurse-submodules https://github.com/swjTheDad/AdaMoE-VLA.git
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```
### 💪 Training
Our training pipeline is identical to openpi, with addtional hyparameters for MoE. Below are new options we provide to adjust your custom openpi model with MoE architectures.
We support the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) benchmark and the [RoboTwin](https://robotwin-platform.github.io/doc/index.html) benchmark. 
```
# We use LIBERO training config as an example here.
_CONFIGS = [
    TrainConfig(
        name="AdaMoE_libero",
        model=pi0.Pi0Config(
            use_moe=True,  # Enable MoE; fall back to original pi0 model if set to False
            moe_type=_gemma.MoEType.ADAMOE,  # Use AdaMoE by default
            num_experts=4, # Number of Routed Experts
            top_k=1, # Top-k value ranging from 1 to num_experts
        ),
        weight_loader=weight_loaders.MoEWeightLoader(
            params_path="path/to/pi0_base/params",
            num_experts=4,  # Set number of routed experts, same as num_experts in the model config above
            noise_std=0.0,  # Set the noise level added to your experts during their initialization; higher noise levels means you will have diverse experts from the beginning, but are more likely to impair the experts' knowledge inherited from the base model
            gating_init_std=0.006,  # Std during router initialization
        ),
        optimizer=_optimizer.MultiGroupAdamW(
            lr_base=2.5e-5,  # Base model components
            lr_moe=2.5e-5,  # MoE components
            lr_router=5e-5,  # Router components
            wd_base=1e-6,  # Base weight decay
            wd_moe=1e-6,  # MoE weight decay
            wd_router=1e-6,  # Router weight decay
            # Cosine decay schedule parameters for each group
            # Warmup steps
            warmup_steps_base=1000,
            warmup_steps_moe=1000,
            warmup_steps_router=100,
            # Decay steps, usually set to number of training steps
            decay_steps_base=90000,
            decay_steps_moe=90000,
            decay_steps_router=90000,
            # Final learning rates
            decay_lr_base=1e-6,
            decay_lr_moe=1e-6,
            decay_lr_router=1e-6,
        ),
    ),
]
```
If you want to train your model on RoboTwin, you can follow their [pi0 document](https://robotwin-platform.github.io/doc/usage/Pi0.html) to generate robotwin data and convert them into [lerobot dataset](https://github.com/huggingface/lerobot) format.
After acquiring RoboTwin data, you need clone our repo into your ```RoboTwin/policy``` directory, set up the virtual environment following previous environment setup steps, and modify ```config.py``` to use the corresponding dataset type:
```
# RoboTwin aloha data for example
data=LeRobotAlohaDataConfig(
    repo_id="your-repo-id", 
    adapt_to_pi=False,
    repack_transforms=_transforms.Group(inputs=[
        _transforms.RepackTransform({
            "images": {
                "cam_high": "observation.images.cam_high",
                "cam_left_wrist": "observation.images.cam_left_wrist",
                "cam_right_wrist": "observation.images.cam_right_wrist",
            },
            "state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
        })
    ]),
    base_config=DataConfig(
        local_files_only=True,  # Set to True for local-only datasets.
        prompt_from_task=True,  # Set to True for prompt by task_name
    ),
),
```
We provide ```finetune.sh``` and ```run.sh``` for convenient finetuning.
First, replace environment variables with your own in ```finetune.sh```:
```
train_config_name=$1
model_name=$2
gpu_use=$3
export OPENPI_DATA_HOME="your_OPENPI_DATA_HOME(checkpoints)"
export HF_LEROBOT_HOME="your_HF_LEROBOT_HOME(datasets)"
export WANDB_MODE=offline # set your W&B mode to offline if you don't want training stats synced right away
export CUDA_VISIBLE_DEVICES=$gpu_use
export XDG_CACHE_HOME="your_XDG_CACHE_HOME" # usually this should be your project root
echo $CUDA_VISIBLE_DEVICES
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 python scripts/train.py $train_config_name --exp-name=$model_name --overwrite
```
Then, replace the project path, experiment name and available GPUs in ```run.sh```:
```
cd "your_project_root_directory"
source .venv/bin/activate
# bash finetune.sh $train_config_name $model_name $gpu_use
bash finetune.sh AdaMoE_libero your_experiments_name 0,1,2,3
```
After the above steps, you can simply run the experiment by:
```
bash run.sh
```

### 📊 Evaluation
The evaluation process on LIBERO is identical to openpi. The evaluation process on RoboTwin is also similar to pi0, except you need to replace a few files. We also provide bash scripts ```test-libero-server.sh``` and ```test-libero-client.sh``` for evaluation.
Evaluation on RoboTwin is identical to the official RoboTwin repo.

## ☎️ Contacts
If you need any assistance, please feel free to raise issues, join our WeChat Group below, or contact the authors directly.
<img width="1161" height="341" src="img/contacts.png" />

## 📝 Citations
If you find our work useful, please consider citing:

Expertise need not monopolize: Action-Specialized Mixture of Experts for Vision-Language-Action Learning
```
@misc{shen2025expertiseneedmonopolizeactionspecialized,
      title={Expertise need not monopolize: Action-Specialized Mixture of Experts for Vision-Language-Action Learning}, 
      author={Weijie Shen and Yitian Liu and Yuhao Wu and Zhixuan Liang and Sijia Gu and Dehui Wang and Tian Nian and Lei Xu and Yusen Qin and Jiangmiao Pang and Xinping Guan and Xiaokang Yang and Yao Mu},
      year={2025},
      eprint={2510.14300},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2510.14300}, 
}
```

## 🥰 Acknowledgement

* RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation ([link](https://robotwin-platform.github.io/doc/index.html))
    <details>
    <summary>Cite RoboTwin 2.0</summary>

    ```
    @article{chen2025robotwin,
    title={Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation},
    author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and Liu, Yibin and Li, Zixuan and Liang, Qiwei and Lin, Xianliang and Ge, Yiheng and Gu, Zhenyu and others},
    journal={arXiv preprint arXiv:2506.18088},
    year={2025}
    }
    ```
    </details>

* Benchmarking Knowledge Transfer for Lifelong Robot Learning ([link](https://libero-project.github.io/main))
    <details>
    <summary>Cite LIBERO</summary>

    ```
    @article{liu2023libero,
    title={LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning},
    author={Liu, Bo and Zhu, Yifeng and Gao, Chongkai and Feng, Yihao and Liu, Qiang and Zhu, Yuke and Stone, Peter},
    journal={arXiv preprint arXiv:2306.03310},
    year={2023}
    }
    ```
    </details>

* $\pi_0$: A Vision-Language-Action Flow Model for General Robot Control ([link](https://github.com/Physical-Intelligence/openpi))
    <details>
    <summary>Cite Openpi$</summary>

    ```
    @misc{black2024pi0visionlanguageactionflowmodel,
        title={$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control}, 
        author={Kevin Black and Noah Brown and Danny Driess and Adnan Esmail and Michael Equi and Chelsea Finn and Niccolo Fusai and Lachy Groom and Karol Hausman and Brian Ichter and Szymon Jakubczak and Tim Jones and Liyiming Ke and Sergey Levine and Adrian Li-Bell and Mohith Mothukuri and Suraj Nair and Karl Pertsch and Lucy Xiaoyang Shi and James Tanner and Quan Vuong and Anna Walling and Haohuan Wang and Ury Zhilinsky},
        year={2024},
        eprint={2410.24164},
        archivePrefix={arXiv},
        primaryClass={cs.LG},
        url={https://arxiv.org/abs/2410.24164}, 
    }
    ```
    </details>
## 🪪 License
This repository is released under the MIT license. See [LICENSE](https://github.com/swjTheDad/AdaMoE-VLA/blob/main/LICENSE) for additional details.
