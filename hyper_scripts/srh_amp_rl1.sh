exp_name=srh_amp_rl1

gpu=1

# for mdl in "iTransformer" "DLinear"; do
#     for dst in "ECL" "Weather"; do
#         for amp in 0.1 0.5 1 5 10 50; do
#             for nfp in 0.05 0.1 0.2; do
#                 job_name=${exp_name}_dst_${dst}_amp_${amp}_nfp_${nfp}_mdl_${mdl}
#                 save_dir="./${exp_name}/${job_name}"
#                 command_args=("${lbd}" "${lr}" "${save_dir}" "${gpu}" "${dtp}")
#                 command_args+=(96 192 336 720)

#                 mkdir -p "${save_dir}/"
#                 (
#                     echo "Running command for $job_name"
#                     bash ./scripts/ICML2024/ltf_srh_amp/${dst}_script/${mdl}.sh "${command_args[@]}"
#                 ) 2>&1 | tee -a "${save_dir}/stdout.txt"
#             done
#         done
#     done
# done



for mdl in "DLinear"; do
    for dst in "ECL"; do
        for nfp in 0.05; do
            for amp in 1 5 10 50; do
                job_name=${exp_name}_amp_$(printf "%g" "$amp")_dst_${dst}_mdl_${mdl}_nfp_${nfp}
                save_dir="./${exp_name}/${job_name}"
                command_args=("${amp}" "${nfp}" "${save_dir}" "${gpu}")

                if [[ $amp == 1 ]]; then
                    command_args+=(720)
                elif [[ $amp == 5 ]]; then
                    command_args+=(96 192 336 720)
                elif [[ $amp == 10 ]]; then
                    command_args+=(192 336)
                else
                    command_args+=(192 720)
                fi

                mkdir -p "${save_dir}/"
                (
                    echo "Running command for $job_name"
                    bash ./scripts/ICML2024/ltf_srh_amp/${dst}_script/${mdl}.sh "${command_args[@]}"
                ) 2>&1 | tee -a "${save_dir}/stdout.txt"
            done
        done


        for nfp in 0.1; do
            for amp in 0.1 0.5 1 5 10 50; do
                job_name=${exp_name}_amp_$(printf "%g" "$amp")_dst_${dst}_mdl_${mdl}_nfp_${nfp}
                save_dir="./${exp_name}/${job_name}"
                command_args=("${amp}" "${nfp}" "${save_dir}" "${gpu}")

                if [[ $amp == 0.1 ]] || [[ $amp == 10 ]]; then
                    command_args+=(96 336 720)
                elif [[ $amp == 0.5 ]] || [[ $amp == 5 ]]; then
                    command_args+=(336 720)
                elif [[ $amp == 1 ]] || [[ $amp == 50 ]]; then
                    command_args+=(96 192 336 720)
                fi

                mkdir -p "${save_dir}/"
                (
                    echo "Running command for $job_name"
                    bash ./scripts/ICML2024/ltf_srh_amp/${dst}_script/${mdl}.sh "${command_args[@]}"
                ) 2>&1 | tee -a "${save_dir}/stdout.txt"
            done
        done


        for nfp in 0.2; do
            for amp in 0.1 0.5 1 5 10 50; do
                job_name=${exp_name}_amp_$(printf "%g" "$amp")_dst_${dst}_mdl_${mdl}_nfp_${nfp}
                save_dir="./${exp_name}/${job_name}"
                command_args=("${amp}" "${nfp}" "${save_dir}" "${gpu}")

                if [[ $amp == 0.1 ]] || [[ $amp == 0.5 ]] || [[ $amp == 1 ]]; then
                    command_args+=(192 720)
                elif [[ $amp == 5 ]]; then
                    command_args+=(96 192 336)
                elif [[ $amp == 10 ]]; then
                    command_args+=(96 192 336 720)
                else
                    command_args+=(336 720)
                fi

                mkdir -p "${save_dir}/"
                (
                    echo "Running command for $job_name"
                    bash ./scripts/ICML2024/ltf_srh_amp/${dst}_script/${mdl}.sh "${command_args[@]}"
                ) 2>&1 | tee -a "${save_dir}/stdout.txt"
            done
        done
    done
done