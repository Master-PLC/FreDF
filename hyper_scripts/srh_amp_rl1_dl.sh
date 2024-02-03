exp_name=srh_amp_rl1

gpu=0

for mdl in "DLinear"; do
    for dst in "Weather" "ECL"; do
        for nfp in 0.1 0.2 0.3; do
            for amp in 10 100 200 500 1000 2000 5000; do
                job_name=${exp_name}_dst_${dst}_amp_${amp}_nfp_${nfp}_mdl_${mdl}
                save_dir="./${exp_name}/${job_name}"
                command_args=("${amp}" "${nfp}" "${save_dir}" "${gpu}")
                command_args+=(336)

                mkdir -p "${save_dir}/"
                (
                    echo "Running command for $job_name"
                    bash ./scripts/ICML2024/ltf_srh_amp/${dst}_script/${mdl}.sh "${command_args[@]}"
                ) 2>&1 | tee -a "${save_dir}/stdout.txt"
            done
        done
    done
done