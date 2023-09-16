# AvatarFusion
Official implementation of ACM Multimedia 2023 paper: 

**AvatarFusion: 
Zero-shot Generation of Clothing-Decoupled 3D Avatars Using 2D Diffusion**
(<a href="https://hansenhuang0823.github.io/AvatarFusion/"> project page </a>)

## To-Do List:
- [ ] open source inference code by Sept. 17th.
- [ ] open source checkpoints of avatars in standing pose by Sept. 28th.
- [✔] open source the training code.
- [ ] open source benchmark FC-50.

## Download Supporting Models:
1. smpl_models/smpl/SMPL_NEUTRAL.pkl: <a href="https://smpl.is.tue.mpg.de/"> https://smpl.is.tue.mpg.de/ </a> (**Note**: please download version 1.0)
2. new_data/gradients_grid_stand.npy, sdf_grid_stand.npy, vertices_stand.npy, bound.npz: <a href="https://drive.google.com/drive/folders/1V1GNMPvbkX6NLC9rcuYjARPtcLjE6-k9?usp=sharing">Google Drive </a>
3. stand_pose.npy: <a href="https://drive.google.com/drive/folders/1V1GNMPvbkX6NLC9rcuYjARPtcLjE6-k9?usp=sharing">Google Drive </a>

## Training

We offer three PSDS schedules, with the default schedule being the most stable one. Since the level of understanding of Stable Diffusion varies among individual characters, if clothing decoupling cannot be achieved, you can switch the schedules by adding "--schedule 1 (or 2)" or modifying the text prompts in the configurations.

    python avatarfusion.py --mode train_diffusion --conf confs/TomCruise_opensource.conf
Low VRAM users can reduce resolution ("max_ray_num") in line 66, avatarfusion.py.

## Acknowledgement
This repository is build upon an increasing list of amazing research works and open-source projects, thanks a lot to all the authors for sharing!

* [DreamFusion: Text-to-3D using 2D Diffusion](https://dreamfusion3d.github.io/)
    ```
    @article{poole2022dreamfusion,
        author = {Poole, Ben and Jain, Ajay and Barron, Jonathan T. and Mildenhall, Ben},
        title = {DreamFusion: Text-to-3D using 2D Diffusion},
        journal = {arXiv},
        year = {2022},
    }
    ```
* [Stable-Dreamfusion](https://github.com/ashawkey/stable-dreamfusion/)
    ```
    @misc{stable-dreamfusion,
        Author = {Jiaxiang Tang},
        Year = {2022},
        Note = {https://github.com/ashawkey/stable-dreamfusion},
        Title = {Stable-dreamfusion: Text-to-3D with Stable-diffusion}
    }
    ```
* [AvatarCLIP: Zero-Shot Text-Driven Generation and Animation of 3D Avatars](https://github.com/hongfz16/AvatarCLIP)
    ```
    @article{hong2022avatarclip,
        title={AvatarCLIP: Zero-Shot Text-Driven Generation and Animation of 3D Avatars},
        author={Hong, Fangzhou and Zhang, Mingyuan and Pan, Liang and Cai, Zhongang and Yang, Lei and Liu, Ziwei},
        journal={ACM Transactions on Graphics (TOG)},
        volume={41},
        number={4},
        articleno={161},
        pages={1--19},
        year={2022},
        publisher={ACM New York, NY, USA},
        doi={10.1145/3528223.3530094},
    }
    ```