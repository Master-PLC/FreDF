exp_name=stf_as_ltf

gpu=5

# for mdl in "iTransformer" "DLinear" "Autoformer" "Transformer" "TimesNet" "PatchTST" "TCN" "LSTM" "Crossformer" "TiDE" "FEDformer" "FreTS"; do
for mdl in "iTransformer"; do
    for lbd in 1.0; do
        for lr in 0.001; do
            job_name=${exp_name}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
            save_dir="./${exp_name}/${job_name}"
            command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}")
            # command_args+=("Yearly" "Quarterly" "Monthly" "Weekly" "Daily" "Hourly")
            command_args+=("Yearly")

            mkdir -p "${save_dir}/"
            (
                echo "Running command for $job_name"
                bash ./scripts/ICML2024/stf_as_ltf/${mdl}.sh "${command_args[@]}"
            ) 2>&1 | tee -a "${save_dir}/stdout.txt"
        done
    done
done

