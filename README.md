# Consistent Depth Estimation for Video

## Quick Start

A very quick guide to using this code: 
1. If freshly cloning this repo, go to the Setup section first.
2. Use the grid_bash_scripts/test_mannequin_on_grid.sh script to test the model. You will need a GPU, which can be specified on the Brown CS Grid or most computers in the Visual Computing Lab. If you are using grid, you can run:
```cd /data/jhtlab/apikieln/logs
qsub -cwd -l gpus=1 -m abes /data/jhtlab/apikieln/mannequinchallenge/grid_bash_scripts/test_mannequin_on_grid.sh```

## Overview
This repo contains the implementation of the Consistent Depth Estimation for Video project, written by Marc Mapeke and Adam Pikielny. Our goal is to improve temporal consistency of depth maps. We start with Google's MannequinChallenge work. Different branches of this repo (TODO, hopefully we will combine these) contain latent regularization, anti-aliased sampling, and fourier features techniques towards this goal. 

Here is a link to my thesis [presentation](https://docs.google.com/presentation/d/1_0Mgygl7-zHsIIfYkqIjycVZTPjHw3pGgcbOI6KdYis/edit#slide=id.g35f391192_00) and [document](https://drive.google.com/file/d/1jF-IOYivDaL0aoN6qJj5AdSrNxRsgSzf/view?usp=sharing) (please request access if necessary, or email me).

I have moved the original README to `README_Google.md`. This README now contains an effort to document our code. 

## Setup

This section was copied from Google's README, now at `README_Google.md`.

The code is based on PyTorch. The code has been tested with PyTorch 1.1 and Python 3.6. 

We recommend setting up a `virtualenv` environment for installing PyTorch and
the other necessary Python packages. The [TensorFlow installation
guide](https://www.tensorflow.org/install/pip) may be helpful (follow steps 1
and 2) or follow the `virtualenv` documentation.

Once your environment is set up and activated, install the necessary packages:

```
(pytorch)$ pip install torch torchvision scikit-image h5py
```

The model checkpoints are stored on Google Cloud and may be retrieved by running:

```
(pytorch)$ ./fetch_checkpoints.sh
```

## Important Files in mannequinchallenge

This is a non-exhaustive list of important python files and folders that we changed and will likely change further.

* `accuracy_scripts/` contains the logic to evaluate MSE accuracy by doing least squares alignment with COLMAP depth. 
* `grid_bash_scripts/` (discussed in Bash Scripts section), scripts to run stuff on grid, such as training, testing, etc. 
* `loaders/`: loads data into model. We have different data loaders for different models. Discussed further in bash scripts section.
* `models/`: very important. This is where the model architecture is defined
    - We mainly modify pix2pixmodel and hourglass. Pix2pix is directly called by the training code (`train_from_scratch.py`). It contains the loss functions, optimizer, etc. The actual model architecture is defined in `hourglass.py`. Hourglass is where we change things like sampling and fourier features. 
    - `resample_v2.py` contains our new sampling filters. The fourier features code is baked directly into `hourglass.py`. 
    - The upsample test files are not actually used by the model but are helpful for testing new sampling filters. 
* `options/train_options.py`: flags to pass when training. Below the line #Added by Adam is stuff that was added this year. See bash scripts for examples of using these flags. 
* `test_davis_videos.py` is the original code to run inference on the model. The "davis" was a reference to the original dataset but this can be modified. 
* `train_from_scratch.py`: We use this for **all training** (it doesn't actually need to be from scratch, if you don't pass the train_from_scratch flag. Sorry this is confusing.). This file includes a data loader, train and test list, and a few other things that can be modified inline. Most things should be modified through flags. The end result of calling this file will be a .pth model saved to `checkpoints/`, along with intermediate inference results (these can be very helpful to see what the model is doing). Bash scripts have examples of how to call this file.

* `test_data/` is where all outputs go, except for the models themselves which go in `checkpoints/`. `test_data/` most importantly includes `viz_predictions`, where depth images are output. The structure varies, but generally it's `viz_predictions -> dataset and/or specific ID -> model_name -> (sometimes) specific epoch of model`. 
Scripts to generate video also output here, and there are also text files containing certain pieces of the datasets. 


## Bash Scripts:
All the bash scripts I wrote to run on grid should be stored in mannequinchallenge/grid_bash_scripts. Preivously, they were free-floating in /data/jhtlab/apikieln, so if something is breaking now it is probably a paths issue.

Some bash scripts are accompanied by a plain text file that shows exactly how to run it. This isn't necessary, but it was helpful for me when switching between scripts. For example, for `test_mannequin_on_grid.sh`, run `cat test_grid_command` to see the command I use, and copy/paste that to trigger `test_mannequin_on_grid.sh` to run on grid. The `-m abes` flag will email grid job status updates. I believe this just emails whoever submitted the job. Tip: I created a filter in my mail app to prevent these emails from flooding my inbox. 

Here are the scripts:
* `test_mannequin_on_grid.sh` and `test_grid_command`: run inference on a given model. Right now, the dataset can't be specified here and must be done manually (TODO I think I have this functionality in other inference functions though?)
* `train_mannequin_on_grid.sh` and `train_grid_command`: train a model. Dataset must still be specified in train_from_scratch.py. (TODO name of python file is confusing because it doesn't always train from scratch.) Other options can be specified on command line, such as epochs, batch size, etc. See `mannequinchallenge/options/train_options.py` for full list of options. 
* `L2_consistency_metric_script.sh` measure consistency of a model. This script doesn't require grid! You can simply call bash L2_... You can specify the data to use. However, this requires that data (output images) already be generated. Also, currently this only measures one video/set of frames at a time, but I think it would be better to measure it for a whole dataset and then take an average. The script outputs an image to `Consistency_Metrics/` showing the difference between each pair of frames. The top has the average difference, which is what we use. 
* `alias_free_L2_consistency_metric_script.sh` TODO I think this script is no longer necessary? 
* `visualize_mannequin_on_grid.sh`: PCA latent visualization of a model. Make sure the proper flags are passed to test_davis_videos.py so that the right model architecture is used when loading! For example, specify if using anti-alias upsampling or downsampling, fourier features, etc.
* `accuracy_metric_script.sh`: measure MSE accuracy compared to COLMAP as ground truth. Because I suspect anti-aliased models generalize better, I was trying with google depth as ground truth, which is in `accuracy_metric_script_google_gt.sh`, but this is still a WIP.
* Generating videos:
    - `convert_frames_to_video_split_screen.sh`: given input frames from multiple models, convert them to a single split screen video. You can also overlay text to label each model. The "3-way" version of this script splits between 3 videos. The "adam_translate" outputs had a different file structure so I ended up duplicating the script, but this isn't really necessary and should be abstracted. Also for these scripts, I would sometimes use a least squares alignment among models first to make them look the same. Right now, this has to be done manually using the accuracy_metric script: `accuracy_metric_google_gt.py --aligning_adam_translate_videos`. This is very hacky and it would be good to abstract this. Aligning them to COLMAP would also generally increase contrast which created better videos. `convert_frames_to_video.sh` works for a single frame set -> video.
    - `generate_result_videos.sh`: This runs inference on a model to generate frames, then converts them to video. This is different from the above which only do the conversion to video step. 

Different configurations of the model require different amounts of memory. For some, the standard 11gb cards on grid are fine. For others, the 24gb card is necessary. Generally, if I run out of memory, I try the same job but decrease the batch size by a factor of 2, or try using the 24gb card if I wasn't already. We also have different data loaders. The Latent data loader loads in pairs of images (to constrain them), so it needs more memory than loading in single images. The data loader can be changed in `train_from_scratch.py`. See `loaders/aligned_data_loader.py` for more info or to change or add a loader. 

## Explanation of folders in /data/jhtlab/apikieln:
These folders are not in this repo (they are at the same level as this repo on jhtlab), but some are important for the project. 

* `venv-mannequin` is the virtual environment that needs to be activated before training or testing our code. It can be activated using
`cd /data/jhtlab/apikieln
source ./venv-mannequin/bin/activate`
* `vrlab_machine_backup` is a backup of previous work we were doing on avd-vrlab3, which is a machine in the visual computing lab. I backed it up because the disk was failing. 
* `checkpoints` contains saved models. This could be moved within the repo. If so, don't push it. 
* `alias-free-torch` is someone else's repo implementing alias-free-gan
* `logs` contains all logs from CS grid jobs. I have a document detailing the purpose of each job I have run, that I can pass along if it would be helpful. In order for a job to output its log here, you must cd into logs before running the job


## Misc
Google's model now outputs depth inversely of ours. This was not originally the case, so we must have changed something but we haven't been able to track it down. For comparison we just invert their depth. 

Overfitting can be very helpful to check if a model is working. We have several small training sets that can be used. Currently in `train_from_scratch.py`, the train and test list are set manually. There is an example of an overfit:

`video_list = 'test_data/small_train_list_grid.txt'
test_video_list = 'test_data/small_test_list_grid.txt'`


## End
Thanks for reading!
Please contact me with any questions!! @AdamPikielny on Slack
