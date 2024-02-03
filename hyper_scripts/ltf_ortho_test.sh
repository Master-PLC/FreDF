exp_name=test

gpu=0
# dst=Traffic
dst=ETTm1

for mdl in "iTransformer"; do
    for lbd in 0.2; do
        for lr in 0.0005; do
            dg_list=(100)

            # for degree in 2 50 100 150 200 250 335; do
            for degree in ${dg_list[@]}; do
                job_name=${exp_name}_dst_${dst}_lbd_$(printf "%g" "$lbd")_lr_$(printf "%g" "$lr")_mdl_${mdl}_dg_${degree}
                save_dir="./${exp_name}/${job_name}"
                command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}" "${degree}")
                command_args+=(192)

                mkdir -p "${save_dir}/"
                (
                    echo "Running command for $job_name"
                    bash ./scripts/ICML2024/ltf_ortho/${mdl}_${dst}.sh "${command_args[@]}"
                ) 2>&1 | tee -a "${save_dir}/stdout.txt"
            done
        done
    done
done