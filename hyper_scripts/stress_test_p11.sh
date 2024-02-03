exp_name=stress_test

gpu=1
# lr=0.0001

for mdl in "iTransformer"; do
    for dst in "ETTh1"; do
        for lr in 0.0001; do
            for dtp in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
                for lbd in 1.0; do
                    job_name=${exp_name}_dst_${dst}_dtp_${dtp}_lbd_$(printf "%g" "$lbd")_mdl_${mdl}_lr_${lr}
                    save_dir="./${exp_name}/${job_name}"
                    command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}" "${dtp}")
                    command_args+=(96 192 336 720)

                    mkdir -p "${save_dir}/"
                    (
                        echo "Running command for $job_name"
                        bash ./scripts/ICML2024/ltf_stress_test/${dst}_script/${mdl}.sh "${command_args[@]}"
                    ) 2>&1 | tee -a "${save_dir}/stdout.txt"
                done
            done
        done
    done
done