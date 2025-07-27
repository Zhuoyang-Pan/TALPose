# `TALPose`: Tracker-Assisted Point Labeling for Pose Annotation in Videos
**[Project page](https://zhuoyang-pan.github.io/animal-labeling/) &bull;
[arXiv](https://arxiv.org/abs/2506.03868)**

`TALPose` is a tracker-assisted labeling tool for annotating keypoints in video sequences. It leverages test-time optimization on general-purpose point trackers to efficiently generate dense pose annotations from sparse labels. This tool was originally developed for animal pose labeling, but it is also applicable to any video sequence where keypoints need to be annotated.

TALPose is based on our paper:
[Animal Pose Labeling Using General-Purpose Point Trackers](https://arxiv.org/abs/2506.03868), presented at CV4Animals@CVPR 2025.

## 🛠️ Workflow

1. **Load a video**  
   Upload your video by clicking the `Choose File` button.  
   – Alternatively, you can try the tool using the sample video in the `sample_videos` directory.

2. **Add keypoints**  
   Click the `NEW` button, then click anywhere on the video frame to place a keypoint.  
   – Repeat this step to add multiple keypoints.

3. **Initial tracking**  
   After adding keypoints, click the `TRACK` button to begin tracking. The tool uses a pre-trained general-purpose tracker ([CoTracker3](https://github.com/facebookresearch/CoTracker3)) to generate initial trajectories for each keypoint.

4. **Interactive refinement**  
   Refine the tracked keypoints by dragging any inaccurate ones to the correct positions.  
   – We recommend providing corrections every 10–20 frames for each keypoint to achieve accurate optimization.  
   – After making corrections, switch the tracking method to `Step 2: Optimize the tracker`, then click `TRACK` again to optimize the trajectories based on your input.

   Additional tips:
   - To delete a keypoint, click the `🗑️` icon.
   - To add more keypoints, click `NEW` and repeat the process.

5. **Save annotations**  
   Once you're satisfied with the keypoints and their trajectories, export your results by saving the annotations in JSON format.

Here is a video demo of the tool:

https://github.com/user-attachments/assets/8fe7f6e0-2a9a-4f33-8526-c64d47550e23


## Installation 
1. Clone the repository and the dependencies
```
git clone https://github.com/Zhuoyang-Pan/TALPose.git --recursive
cd TALPose
```

2. Download the pre-trained model of cotracker
```
mkdir -p dependencies/cotracker3/checkpoints
wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth -O dependencies/cotracker3/checkpoints/scaled_offline.pth
```

2. First install torch following the instructions at https://pytorch.org/get-started/locally/
Then install other Python dependencies
```
pip install -r requirements.txt
```
If you don't have node.js installed, you can install it(we recommend v22.16.0) from https://nodejs.org/en/download/, and then install the JavaScript dependencies
```
npm install
```

## Usage
1. Run the server
```
uvicorn main:app --reload
```
2. Run the client
```
npm start
```

## Status

_July 22, 2025_: Initial release

The tool is in active development, and we are working on improving the user experience and adding more features.

- [ ] Examples and detailed documentation for using the tool.
- [ ] Add post-processing methods(filters) to further refine the keypoints.
- [ ] Load and save annotations in different formats.

## Citation

This codebase is released with the following paper.

<table><tr><td>
    Zhuoyang Pan<sup>1, 2</sup>, Boxiao Pan<sup>1</sup>, Guandao Yang<sup>1</sup>, Adam W. Harley<sup>1</sup>, Leonidas Guibas<sup>1</sup>.
    <strong>Animal Pose Labeling Using General-Purpose Point Trackers</strong>.
    CV4Animals@CVPR 2025, Oral Presentation.
</td></tr>
</table>
<sup>1</sup><em>Stanford University</em>, <sup>2</sup><em>ShanghaiTech University</em>

Please cite our paper if you find this work useful for your research:

```
@article{pan2025animal,
  title     = {Animal Pose Labeling Using General-Purpose Point Trackers},
  author    = {Pan, Zhuoyang and Pan, Boxiao and Yang, Guandao and Harley, Adam W and Guibas, Leonidas},
  journal   = {arXiv preprint arXiv:2506.03868},
  year      = {2025}
}
```

Thanks!
