exp_name=imp_full
out_root="${exp_name}"

lr=0.0005
gpu=1

for dst in "ECL"; do
    for mdl in "iTransformer"; do
        # for lr in 0.0005 0.001 0.002; do
        #     for lbd in 0.0 0.2 0.4 0.6 0.8; do
        #         job_name=${exp_name}_dst_${dst}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
        #         save_dir="./${exp_name}/${job_name}"
        #         command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}")
        #         command_args+=(0.125 0.25 0.375 0.5)

        #         mkdir -p "${save_dir}/"
        #         (
        #             echo "Running command for $job_name"
        #             bash ./scripts/ICML2024/imp_full/${dst}_script/${mdl}.sh "${command_args[@]}"
        #         ) 2>&1 | tee -a "${save_dir}/stdout.txt"
        #     done
        # done


        for lr in 0.0005; do
            for lbd in 0.4 0.6 0.8; do
                job_name=${exp_name}_dst_${dst}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
                save_dir="./${exp_name}/${job_name}"
                command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}")
                if [[ $lbd == 0.4 ]]; then
                    command_args+=(0.25)
                elif [[ $lbd == 0.6 ]]; then
                    command_args+=(0.125)
                else
                    command_args+=(0.25)
                fi

                mkdir -p "${save_dir}/"
                (
                    echo "Running command for $job_name"
                    bash ./scripts/ICML2024/imp_full/${dst}_script/${mdl}.sh "${command_args[@]}"
                ) 2>&1 | tee -a "${save_dir}/stdout.txt"
            done
        done


        for lr in 0.001; do
            for lbd in 0.0 0.2; do
                job_name=${exp_name}_dst_${dst}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
                save_dir="./${exp_name}/${job_name}"
                command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}")
                if [[ $lbd == 0.0 ]]; then
                    command_args+=(0.125)
                elif [[ $lbd == 0.2 ]]; then
                    command_args+=(0.125)
                fi

                mkdir -p "${save_dir}/"
                (
                    echo "Running command for $job_name"
                    bash ./scripts/ICML2024/imp_full/${dst}_script/${mdl}.sh "${command_args[@]}"
                ) 2>&1 | tee -a "${save_dir}/stdout.txt"
            done
        done

        
        for lr in 0.002; do
            for lbd in 0.0 0.6 0.8; do
                job_name=${exp_name}_dst_${dst}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
                save_dir="./${exp_name}/${job_name}"
                command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}")
                if [[ $lbd == 0.0 ]]; then
                    command_args+=(0.5)
                else
                    command_args+=(0.125 0.25 0.375 0.5)
                fi

                mkdir -p "${save_dir}/"
                (
                    echo "Running command for $job_name"
                    bash ./scripts/ICML2024/imp_full/${dst}_script/${mdl}.sh "${command_args[@]}"
                ) 2>&1 | tee -a "${save_dir}/stdout.txt"
            done
        done
    done
done





# exp_name=test
# out_root="${exp_name}"

# lr=0.0005
# gpu=0

# for dst in "ETTm1"; do
#     # for mdl in "TimesNet" "PatchTST" "TCN" "LSTM" "Crossformer" "TiDE" "FEDformer" "Koopa" "FreTS"; do
#     for mdl in "FreTS"; do
#         for lbd in 1.0; do
#             job_name=${exp_name}_dst_${dst}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
#             save_dir="./${out_root}/${job_name}"
#             command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}")
#             command_args+=(96 192 336 720)

#             mkdir -p "${save_dir}/"
#             (
#                 echo "Running command for $job_name"
#                 bash ./scripts/ICML2024/ltf_srh_lambda_lr_large_ot_supp/${dst}_script/${mdl}.sh "${command_args[@]}"
#             ) 2>&1 | tee -a "${save_dir}/stdout.txt"
#         done
#     done
# done