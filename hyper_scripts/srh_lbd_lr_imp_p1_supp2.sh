exp_name=srh_lbd_lr_imp

gpu=2

# for mdl in "iTransformer"; do
#     for dst in "ETTh1"; do
#         for lr in 0.002 0.005; do
#             for lbd in 0.2 0.4 0.6 0.8 1.0 0.0; do
#                 job_name=${exp_name}_dst_${dst}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
#                 save_dir="./${exp_name}/${job_name}"
#                 command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}")
#                 command_args+=(0.125 0.25 0.375 0.5)

#                 mkdir -p "${save_dir}/"
#                 (
#                     echo "Running command for $job_name"
#                     bash ./scripts/ICML2024/imp_srh_lambda_lr/${dst}_script/${mdl}.sh "${command_args[@]}"
#                 ) 2>&1 | tee -a "${save_dir}/stdout.txt"
#             done
#         done
#     done
# done


# for mdl in "iTransformer"; do
#     for dst in "ETTh1"; do
#         for lr in 0.002; do
#             for lbd in 0.6; do
#                 job_name=${exp_name}_dst_${dst}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
#                 save_dir="./${exp_name}/${job_name}"
#                 command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}")
#                 command_args+=(0.125 0.375)

#                 mkdir -p "${save_dir}/"
#                 (
#                     echo "Running command for $job_name"
#                     bash ./scripts/ICML2024/imp_srh_lambda_lr/${dst}_script/${mdl}.sh "${command_args[@]}"
#                 ) 2>&1 | tee -a "${save_dir}/stdout.txt"
#             done
#         done
#     done
# done


for mdl in "DLinear"; do
    for dst in "ETTh1"; do
        for lr in 0.001; do
            for lbd in 0.0 0.2 0.4 0.6 0.8; do
                job_name=${exp_name}_dst_${dst}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
                save_dir="./${exp_name}/${job_name}"
                command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}")
                command_args+=(0.125 0.25 0.375 0.5)

                mkdir -p "${save_dir}/"
                (
                    echo "Running command for $job_name"
                    bash ./scripts/ICML2024/imp_srh_lambda_lr/${dst}_script/${mdl}.sh "${command_args[@]}"
                ) 2>&1 | tee -a "${save_dir}/stdout.txt"
            done
        done
    done
done