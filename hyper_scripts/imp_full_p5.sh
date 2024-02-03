exp_name=imp_full
out_root="${exp_name}"

lr=0.0005
gpu=2

for dst in "ECL"; do
    # for mdl in "iTransformer" "DLinear" "Autoformer" "Transformer" "TimesNet" "PatchTST" "TCN" "LSTM" "Crossformer" "TiDE" "FEDformer"; do
    for mdl in "TimesNet" "PatchTST" "TCN" "LSTM" "Crossformer" "TiDE" "FEDformer"; do
        if [[ $mdl == 'iTransformer' ]]; then
            lr=0.001
        else
            lr=0.0005
        fi

        for lbd in 1.0; do
            job_name=${exp_name}_dst_${dst}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
            save_dir="./${exp_name}/${job_name}"
            command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}")
            command_args+=(0.125 0.25 0.375 0.5)

            mkdir -p "${save_dir}/"
            (
                echo "Running command for $job_name"
                bash ./scripts/ICML2024/imp_full/${dst}_script/${mdl}.sh "${command_args[@]}"
            ) 2>&1 | tee -a "${save_dir}/stdout.txt"
        done
    done
done



# exp_name=test
# out_root="${exp_name}"

# lr=0.0005
# gpu=2

# for dst in "ECL"; do
#     for mdl in "iTransformer" "DLinear" "Autoformer" "Transformer" "TimesNet" "PatchTST" "TCN" "LSTM" "Crossformer" "TiDE" "FEDformer"; do
#     for mdl in "iTransformer"; do
#         if [[ $mdl == 'iTransformer' ]]; then
#             lr=0.001
#         else
#             lr=0.0005
#         fi

#         for lbd in 1.0; do
#             job_name=${exp_name}_dst_${dst}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
#             save_dir="./${exp_name}/${job_name}"
#             command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}")
#             command_args+=(0.125 0.25 0.375 0.5)

#             mkdir -p "${save_dir}/"
#             (
#                 echo "Running command for $job_name"
#                 bash ./scripts/ICML2024/imp_full/${dst}_script/${mdl}.sh "${command_args[@]}"
#             ) 2>&1 | tee -a "${save_dir}/stdout.txt"
#         done
#     done
# done