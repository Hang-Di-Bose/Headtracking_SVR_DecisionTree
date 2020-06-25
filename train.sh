#!/bin/bash
# trail_local.sh
# A simple shell script to run template locally
# Other than running full jobs locally, the ability
# to do so if important for debugging your code before submitting a job to google cloud ml
# baseline (normalize on full session)
python ./DecisionTree.py \
--output_dir="/c/Users/HD1047208/OneDrive - Bose Corporation/Desktop/models/HeadTracking_SVR/cp" \
--normalization="baseline" \
--input_dir="/c/Users/HD1047208/PycharmProjects/data_3/" \
--path_to_recording_session="/c/Users/HD1047208/OneDrive - Bose Corporation/Desktop/github/" \
--model_type="seq_2_seq_svr" \
--num_epochs=1 \
--batch_size=128 \
--optimizer='adam' \
--loss='mean_squared_error' \
--callback_list='best_checkpoint,checkpoint,tensorboard,early' \
--eval_metrics='mse' \
--sample_rate=100 \
--input_window_length_ms=500 \
--output_window_length_ms=30 \
--window_hop_ms=30 \
--num_signals=4 \
--write_csv=t
