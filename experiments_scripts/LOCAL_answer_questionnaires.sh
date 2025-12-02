#!/bin/bash
module --force purge

source ../deep_steering_env/bin/activate # this relies on the symlink in the root project folder
nvidia-smi  # Double check the allocated GPUs

cd ../src

# Qwen (CHAT - Instruction-Tuned)
# qwen3-0.6b-chat qwen3-1.7b-chat qwen3-4b-chat qwen3-8b-chat qwen3-14b-chat |||| qwen3-32b-chat qwen3-30b-a3b-chat

for model in qwen3-4b-base qwen3-8b-base qwen3-0.6b-chat qwen3-1.7b-chat qwen3-4b-chat qwen3-8b-chat qwen3-14b-chat
do
    for permutation_strategy in random reverse
    do
        for system_prompt in auth_left auth_right lib_left lib_right generic centrist
        do
            python ./answer_questionnaire.py -q political_compass -m $model -ps $permutation_strategy -sp $system_prompt -b 16
        done
    done
done

# Analyse results and calculate scores
python ./questionnaires_analysis/analyse_political_compass_results.py