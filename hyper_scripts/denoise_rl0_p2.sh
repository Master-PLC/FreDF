exp_name=denoise

gpu=3
ax=1.0


for mdl in "iTransformer"; do
    for dst in "ECL"; do
        for nfp in 0.5; do
            amp_list=(1000)
            for amp in "${amp_list[@]}"; do
                job_name=${exp_name}_dst_${dst}_amp_$(printf "%g" "$amp")_nfp_${nfp}_mdl_${mdl}_ax_$(printf "%g" "$ax")
                save_dir="./${exp_name}/${job_name}"
                command_args=("${ax}" "${amp}" "${nfp}" "${gpu}" "${save_dir}")
                command_args+=(192 336 720)

                mkdir -p "${save_dir}/"
                (
                    echo "Running command for $job_name"
                    bash ./scripts/ICML2024/denoise_rl0/${dst}_script/${mdl}.sh "${command_args[@]}"
                ) 2>&1 | tee -a "${save_dir}/stdout.txt"
            done
        done

        for nfp in 0.25 0.125 0.05; do
            amp_list=(500 1000)
            for amp in "${amp_list[@]}"; do
                job_name=${exp_name}_dst_${dst}_amp_$(printf "%g" "$amp")_nfp_${nfp}_mdl_${mdl}_ax_$(printf "%g" "$ax")
                save_dir="./${exp_name}/${job_name}"
                command_args=("${ax}" "${amp}" "${nfp}" "${gpu}" "${save_dir}")
                command_args+=(96 192 336 720)

                mkdir -p "${save_dir}/"
                (
                    echo "Running command for $job_name"
                    bash ./scripts/ICML2024/denoise_rl0/${dst}_script/${mdl}.sh "${command_args[@]}"
                ) 2>&1 | tee -a "${save_dir}/stdout.txt"
            done
        done
    done
done


