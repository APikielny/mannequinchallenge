Bash Scripts:
All bash scripts I wrote to run on grid should be stored in mannequinchallenge/grid_bash_scripts. Preivously, they were free-floating in /data/jhtlab/apikieln, so if something is breaking now it probably is a paths issue.

Some scripts (.sh) are accompanied by a plain text file that shows exactly how to run it. This isn't necessary, but it was helpful for me when switching betwee scripts. For example, for `test_mannequin_on_grid.sh`, run `cat test_grid_command` to see the command I use, and copy/paste that to trigger `test_mannequin_on_grid.sh` to run on grid. 

Here are the scripts:
* `test_mannequin_on_grid.sh` and `test_grid_command`: run inference on a given model. Right now, the dataset can't be specified here and must be done manually (TODO I think I have this functionality in other inference functions though?)
* `train_mannequin_on_grid.sh` and `train_grid_command`: train a model. Dataset must still be specified in train_from_scratch.py. (TODO name of python file is confusing because it doesn't always train from scratch.) Other options can be specified on command line, such as epochs, batch size, etc. See `mannequinchallenge/options/train_options.py` for full list of options. 
* `L2_consistency_metric_script.sh` measure consistency of a model. This script doesn't require grid to use! You can simply call bash L2_... You can specify the data to use. However, this requires that data (output images) already be generated. Also, currently this only measures one video/set of frames at a time, but I think it would be better to measure it for a whole dataset and then take an average. 
* `alias_free_L2_consistency_metric_script.sh` TODO I think this script is no longer necessary? 


Explanation of folders in /data/jhtlab/apikieln:
* venv-mannequin is the virtual environment that needs to be activated before training or testing our code. It can be activated using
`cd /data/jhtlab/apikieln
source ./venv-mannequin/bin/activate`
* vrlab_machine_backup is a backup of previous work we were doing on avd-vrlab3, which is a machine in the visual computing lab. I backed it up because the disk was failing. 
* checkpoints contains saved models
* alias-free-torch is someone else's repo implementing alias-free-gan
* logs contains all logs from CS grid jobs. I have a document detailing the purpose of each job. In order for a job to output its log here, you must cd into logs before running the job
* 