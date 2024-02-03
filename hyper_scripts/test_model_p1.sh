exp_name=srh_lbd_lr_large
out_root="${exp_name}"

gpu=0

for dst in "ETTm2"; do
    for mdl in "iTransformer" "Autoformer" "DLinear"; do
        for lbd in 1.0; do
            lr=0.0005
            job_name=${exp_name}_dst_${dst}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}
            save_dir="./${out_root}/${job_name}"
            command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}")
            command_args+=(96 192 336 720)

            mkdir -p "${save_dir}/"
            (
                echo "Running command for $job_name"
                bash ./scripts/ICML2024/test_ltf/${dst}_script/${mdl}.sh "${command_args[@]}"
            ) 2>&1 | tee -a "${save_dir}/stdout.txt"
        done
    done
done