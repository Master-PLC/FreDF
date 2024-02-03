exp_name=stf_large_lbd

gpu=2

# for mdl in "iTransformer"; do
#     for lbd in 0.0; do
#         for lr in 0.00005 0.00002 0.00001; do
#             job_name=${exp_name}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
#             save_dir="./${exp_name}/${job_name}"
#             command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}")
#             # command_args+=("Yearly" "Quarterly" "Monthly" "Weekly" "Daily" "Hourly")
#             command_args+=("Yearly")

#             mkdir -p "${save_dir}/"
#             (
#                 echo "Running command for $job_name"
#                 bash ./scripts/ICML2024/stf_srh_lambda_lr/${mdl}_M4.sh "${command_args[@]}"
#             ) 2>&1 | tee -a "${save_dir}/stdout.txt"
#         done
#     done
# done


# for mdl in "iTransformer"; do
#     for lbd in 0.0; do
#         # for lr in 0.00005 0.00002 0.00001; do
#         # for lr in 0.005 0.01 0.02 0.05; do
#         for lr in 0.08 0.1 0.15 0.2; do
#             job_name=${exp_name}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
#             save_dir="./${exp_name}/${job_name}"
#             command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}")
#             # command_args+=("Yearly" "Quarterly" "Monthly" "Weekly" "Daily" "Hourly")
#             command_args+=("Yearly")

#             mkdir -p "${save_dir}/"
#             (
#                 echo "Running command for $job_name"
#                 bash ./scripts/ICML2024/stf_srh_lambda_lr/${mdl}_M4.sh "${command_args[@]}"
#             ) 2>&1 | tee -a "${save_dir}/stdout.txt"
#         done
#     done
# done



for mdl in "iTransformer"; do
    # for lbd in 0.85 0.9 0.95 0.99 0.995 0.999 0.9995 0.9999; do
    for lbd in 0.995 0.999 0.9995 0.9999; do
        # for lr in 0.0005 0.0002 0.001 0.002; do
        # for lr in 0.0005 0.0002 0.001 0.002; do
        # for lr in 0.005 0.01 0.02 0.05; do
        for lr in 0.1 0.2 0.15; do
            job_name=${exp_name}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
            save_dir="./${exp_name}/${job_name}"
            command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}")
            # command_args+=("Yearly" "Quarterly" "Monthly" "Weekly" "Daily" "Hourly")
            command_args+=("Yearly" "Monthly" "Weekly" "Daily" "Hourly")
            # command_args+=("Quarterly")

            mkdir -p "${save_dir}/"
            (
                echo "Running command for $job_name"
                bash ./scripts/ICML2024/stf_srh_lambda_lr/${mdl}_M4.sh "${command_args[@]}"
            ) 2>&1 | tee -a "${save_dir}/stdout.txt"
        done
    done
done



# for mdl in "iTransformer"; do
#     for lbd in 0.995 0.999 0.9995 0.9999; do
#         for lr in 0.005 0.01 0.02 0.05; do
#             job_name=${exp_name}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
#             save_dir="./${exp_name}/${job_name}"
#             command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}")
#             # command_args+=("Yearly" "Quarterly" "Monthly" "Weekly" "Daily" "Hourly")
#             # command_args+=("Yearly" "Monthly" "Weekly" "Daily" "Hourly")
#             command_args+=("Quarterly")

#             mkdir -p "${save_dir}/"
#             (
#                 echo "Running command for $job_name"
#                 bash ./scripts/ICML2024/stf_srh_lambda_lr/${mdl}_M4.sh "${command_args[@]}"
#             ) 2>&1 | tee -a "${save_dir}/stdout.txt"
#         done
#     done
# done