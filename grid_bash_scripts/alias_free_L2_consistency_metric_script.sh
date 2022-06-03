#This script calculates the L2 blur variance for a given set of test frames. Model_name is where the frames are. Epoch num is optional, for the structure model_name/epoch_[epoch_num]/

model_name=${1:-""}
model_data=${3:-"adam_translate"}
epoch_num=${2:-""}

cd /data/jhtlab/apikieln
source ./venv-mannequin/bin/activate

cd alias-free-mannequinchallenge
echo "evaluating consistency metrics..."
mkdir -p Consistency_Metrics/${model_data}
echo "attempting to measure consistency of a single model"
if [[ ${epoch_num} == "" ]]
then
python3 metrics.py --L2_folder test_data/viz_predictions/${model_data}/${model_name} 
else
python3 metrics.py --L2_folder test_data/viz_predictions/${model_data}/${model_name} --epoch ${epoch_num} > /dev/null

fi
#echo "attempting to evaluate consistency across epochs"
#python3 metrics.py --L2_folder test_data/viz_predictions/${model_data}/${model_name} --consistency_over_time #> /dev/null

echo "metrics written out to L2_frame_comparisons/${model_data}/${model_name}_L2_plot.png"
