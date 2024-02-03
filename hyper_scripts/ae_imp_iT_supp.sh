exp_name=ae_imp_supp

gpu=1
num_proc=2

for mdl in "iTransformer"; do
    for dst in "ETTh1" "ETTm1"; do
        for lbd in 0.0 0.2 0.4 0.6 0.8 1.0; do
            for lr in 0.0005 0.0002 0.001; do
                job_name=${exp_name}_dst_${dst}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
                save_dir="./${exp_name}/${job_name}"
                command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}" "${num_proc}")
                command_args+=(0.25)

                mkdir -p "${save_dir}/"
                (
                    echo "Running command for $job_name"
                    bash ./scripts/ICML2024/imp_ae/${dst}_script/${mdl}.sh "${command_args[@]}"
                ) 2>&1 | tee -a "${save_dir}/stdout.txt"
            done
        done
    done
done
