exp_name=srh_lbd_lr_large
out_root="${exp_name}_others"

lr=0.0005
gpu=1

for dst in "ETTh1"; do
    # for mdl in "TimesNet" "PatchTST" "TCN" "LSTM" "Crossformer" "TiDE" "FEDformer" "FreTS"; do
    #     for lbd in 1.0; do
    #         job_name=${exp_name}_dst_${dst}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
    #         save_dir="./${out_root}/${job_name}"
    #         command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}")
    #         command_args+=(96 192 336 720)

    #         mkdir -p "${save_dir}/"
    #         (
    #             echo "Running command for $job_name"
    #             bash ./scripts/ICML2024/ltf_srh_lambda_lr_large_ot_supp/${dst}_script/${mdl}.sh "${command_args[@]}"
    #         ) 2>&1 | tee -a "${save_dir}/stdout.txt"
    #     done
    # done


    for mdl in "TiDE" "FEDformer" "FreTS"; do
        for lbd in 1.0; do
            job_name=${exp_name}_dst_${dst}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
            save_dir="./${out_root}/${job_name}"
            command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}")
            if [[ $mdl == "TiDE" ]]; then
                command_args+=(336 720)
            else
                command_args+=(96 192 336 720)
            fi

            mkdir -p "${save_dir}/"
            (
                echo "Running command for $job_name"
                bash ./scripts/ICML2024/ltf_srh_lambda_lr_large_ot_supp/${dst}_script/${mdl}.sh "${command_args[@]}"
            ) 2>&1 | tee -a "${save_dir}/stdout.txt"
        done
    done
done





# exp_name=test
# out_root="${exp_name}"

# lr=0.0005
# gpu=1

# for dst in "ETTh1"; do
#     # for mdl in "TimesNet" "PatchTST" "TCN" "LSTM" "Crossformer" "TiDE" "FEDformer" "FreTS"; do
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