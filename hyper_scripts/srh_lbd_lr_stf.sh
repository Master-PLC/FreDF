exp_name=srh_lbd_lr_stf

gpu=0

for mdl in "iTransformer"; do
    for lbd in 0.0 0.2 0.4 0.6 0.8 1.0; do
        for lr in 0.0002 0.0005 0.001; do
            job_name=${exp_name}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
            save_dir="./${exp_name}/${job_name}"
            command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}")
            command_args+=("Yearly" "Quarterly" "Monthly" "Weekly" "Daily" "Hourly")

            mkdir -p "${save_dir}/"
            (
                echo "Running command for $job_name"
                bash ./scripts/ICML2024/stf_srh_lambda_lr/${mdl}_M4.sh "${command_args[@]}"
            ) | tee "${save_dir}/stdout.txt"
        done
    done
done


for mdl in "Autoformer" "DLinear" "Transformer"; do
    for lbd in 0.0 0.2 0.4 0.6 0.8 1.0; do
        for lr in 0.001; do
            job_name=${exp_name}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
            save_dir="./${exp_name}/${job_name}"
            command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}")
            command_args+=("Yearly" "Quarterly" "Monthly" "Weekly" "Daily" "Hourly")

            mkdir -p "${save_dir}/"
            (
                echo "Running command for $job_name"
                bash ./scripts/ICML2024/stf_srh_lambda_lr/${mdl}_M4.sh "${command_args[@]}"
            ) | tee "${save_dir}/stdout.txt"
        done
    done
done