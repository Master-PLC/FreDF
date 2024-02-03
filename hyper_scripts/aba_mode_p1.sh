exp_name=aba_mode

gpu=0
lr=0.001

for dst in "ECL"; do
    for mdl in "iTransformer"; do
        for lbd in 0.01 0.05 0.15 0.2 0.005; do
            for mode in "mag" "phase"; do
                job_name=${exp_name}_dst_${dst}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}_mode_${mode}
                save_dir="./${exp_name}/${job_name}"
                command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}" "${mode}")
                command_args+=(96 192 336 720)

                mkdir -p "${save_dir}/"
                (
                    echo "Running command for $job_name"
                    bash ./scripts/ICML2024/aba_mode/${mdl}_${dst}.sh "${command_args[@]}"
                ) 2>&1 | tee -a "${save_dir}/stdout.txt"
            done
        done
    done
done