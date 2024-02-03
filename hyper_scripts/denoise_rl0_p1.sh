exp_name=denoise

gpu=2
ax=1.0


for mdl in "iTransformer"; do
    for dst in "ECL"; do
        # for nfp in 0.5; do
        #     amp_list=(100 200)
        #     for amp in "${amp_list[@]}"; do
        #         job_name=${exp_name}_dst_${dst}_amp_$(printf "%g" "$amp")_nfp_${nfp}_mdl_${mdl}_ax_$(printf "%g" "$ax")
        #         save_dir="./${exp_name}/${job_name}"
        #         command_args=("${ax}" "${amp}" "${nfp}" "${gpu}" "${save_dir}")
        #         if [[ $amp == 100 ]]; then
        #             command_args+=(192 336 720)
        #         else
        #             command_args+=(96 192 336 720)
        #         fi

        #         mkdir -p "${save_dir}/"
        #         (
        #             echo "Running command for $job_name"
        #             bash ./scripts/ICML2024/denoise_rl0/${dst}_script/${mdl}.sh "${command_args[@]}"
        #         ) 2>&1 | tee -a "${save_dir}/stdout.txt"
        #     done
        # done

        for nfp in 0.125 0.05; do
            amp_list=(10 100 200)
            for amp in "${amp_list[@]}"; do
                job_name=${exp_name}_dst_${dst}_amp_$(printf "%g" "$amp")_nfp_${nfp}_mdl_${mdl}_ax_$(printf "%g" "$ax")
                save_dir="./${exp_name}/${job_name}"
                command_args=("${ax}" "${amp}" "${nfp}" "${gpu}" "${save_dir}")
                if [[ $nfp == 0.125 ]] && [[ $amp == 10 ]]; then
                    command_args+=(336 720)
                else
                    command_args+=(96 192 336 720)
                fi

                mkdir -p "${save_dir}/"
                (
                    echo "Running command for $job_name"
                    bash ./scripts/ICML2024/denoise_rl0/${dst}_script/${mdl}.sh "${command_args[@]}"
                ) 2>&1 | tee -a "${save_dir}/stdout.txt"
            done
        done
    done
done
