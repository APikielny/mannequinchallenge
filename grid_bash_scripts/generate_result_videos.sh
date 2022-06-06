############
#run inference for a given model/dataset, and create videos (based on accuracy_metric_script.sh)
############
dataset=${1:-"brown-campus-data"}
model_name=${2:-"best_depth_Ours_Bilinear_inc_3"}
manequin_repo=${4:-"/data/jhtlab/apikieln/mannequinchallenge"}
mannequin_dataset_path=${5:-"/data/jhtlab/apikieln/mannequin-dataset"}

cd /data/jhtlab/apikieln
source ./venv-mannequin/bin/activate
cd ${mannequin_dataset_path}
cd ${dataset}


#for every id in the dataset, check if there has already been inference on this dataset for this model
#if not yet generated, generate using test_davis_videos.py
    #generate test_list

    #test using test list and put in folder corresponding to ID and data-set

# echo "Generating txt files for each ID in dataset"

echo "############"
echo "Generating test data for model " ${model_name}
echo "############"
text_files_generated=0
id_inferences=0
total_ids=0
for d in */ ; do
    id_long=$d
    id=${id_long::-1} #chop slash from end of ID

    #### generate txt file
    txt_file=${manequin_repo}/test_data/txt_files/${dataset}/${id}.txt
    # if ! test -f "$txt_file"; then
    #     cd ${manequin_repo}/accuracy_scripts
    #     python generate_txt_list.py --id ${id} --dataset ${dataset}
    #     cd ${mannequin_dataset_path}/${dataset}
    #     text_files_generated=$(($text_files_generated + 1))
    # fi
    # total_ids=$(($total_ids + 1))

    #### run inference
    #need to specify ID, dataset, model
    cd ${manequin_repo}
    tst_folder=${manequin_repo}/test_data/viz_predictions/generating_result_videos/${dataset}/${id}/images/${model_name}
    if [ ! -d ${tst_folder} ]; then
        echo "Running inference for id " ${id}
        python accuracy_test_videos.py --viz_folder "generating_result_videos" --accuracy_test_list ${txt_file} --weights ${model_name} --accuracy_id ${id} --accuracy_dataset ${dataset} #>/dev/null 2>&1
        id_inferences=$(($id_inferences + 1))
    fi

    echo "Converting frames to video for id " ${id}
    bash convert_frames_to_video.sh ${model_name} ${dataset} ${id}
done
echo "#############"
echo "Finished generating data."
echo "Found ${total_ids} ids. Generated ${text_files_generated} text files and ran inference on ${id_inferences} ids."
echo "#############"

echo "Script has completed."
