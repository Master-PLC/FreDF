exp_name=srh_ax_amp_rl0

gpu=0
ax=1.0

# for mdl in "iTransformer" "DLinear"; do
#     for dst in "ECL" "Weather"; do
#         for nfp in 0.05 0.1 0.2; do
#             for amp in 0.1 0.5 1 5 10 50; do
#                 job_name=${exp_name}_amp_$(printf "%g" "$amp")_ax_$(printf "%g" "$ax")_dst_${dst}_mdl_${mdl}_nfp_${nfp}
#                 save_dir="./${exp_name}/${job_name}"
#                 command_args=("${ax}" "${amp}" "${nfp}" "${gpu}" "${save_dir}")
#                 command_args+=(96 192 336 720)

#                 mkdir -p "${save_dir}/"
#                 (
#                     echo "Running command for $job_name"
#                     bash ./scripts/ICML2024/ltf_srh_ax_amp/${dst}_script/${mdl}.sh "${command_args[@]}"
#                 ) 2>&1 | tee -a "${save_dir}/stdout.txt"
#             done
#         done
#     done
# done


# for mdl in "iTransformer"; do
#     for dst in "Weather"; do
#         for nfp in 0.1; do
#             for amp in 1 50; do
#                 job_name=${exp_name}_amp_$(printf "%g" "$amp")_ax_$(printf "%g" "$ax")_dst_${dst}_mdl_${mdl}_nfp_${nfp}
#                 save_dir="./${exp_name}/${job_name}"
#                 command_args=("${ax}" "${amp}" "${nfp}" "${gpu}" "${save_dir}")
#                 if [[ $amp == 1 ]]; then
#                     command_args+=(720)
#                 else
#                     command_args+=(336 720)
#                 fi

#                 mkdir -p "${save_dir}/"
#                 (
#                     echo "Running command for $job_name"
#                     bash ./scripts/ICML2024/ltf_srh_ax_amp/${dst}_script/${mdl}.sh "${command_args[@]}"
#                 ) 2>&1 | tee -a "${save_dir}/stdout.txt"
#             done
#         done
#     done
# done



for mdl in "DLinear"; do
    for dst in "ECL"; do
        for nfp in 0.05; do
            for amp in 0.1 0.5 1 5 10 50; do
                job_name=${exp_name}_amp_$(printf "%g" "$amp")_ax_$(printf "%g" "$ax")_dst_${dst}_mdl_${mdl}_nfp_${nfp}
                save_dir="./${exp_name}/${job_name}"
                command_args=("${ax}" "${amp}" "${nfp}" "${gpu}" "${save_dir}")
                
                if [[ $amp == 0.1 ]] || [[ $amp == 10 ]]; then
                    command_args+=(192 720)
                elif [[ $amp == 0.5 ]] || [[ $amp == 5 ]]; then
                    command_args+=(96 192 720)
                elif [[ $amp == 1 ]]; then
                    command_args+=(96 720)
                else
                    command_args+=(96 336)
                fi

                mkdir -p "${save_dir}/"
                (
                    echo "Running command for $job_name"
                    bash ./scripts/ICML2024/ltf_srh_ax_amp/${dst}_script/${mdl}.sh "${command_args[@]}"
                ) 2>&1 | tee -a "${save_dir}/stdout.txt"
            done
        done


        for nfp in 0.1; do
            for amp in 0.1 0.5 1; do
                job_name=${exp_name}_amp_$(printf "%g" "$amp")_ax_$(printf "%g" "$ax")_dst_${dst}_mdl_${mdl}_nfp_${nfp}
                save_dir="./${exp_name}/${job_name}"
                command_args=("${ax}" "${amp}" "${nfp}" "${gpu}" "${save_dir}")
                if [[ $amp == 0.1 ]] || [[ $amp == 50 ]]; then
                    command_args+=(96 336)
                elif [[ $amp == 0.5 ]] || [[ $amp == 1 ]]; then
                    command_args+=(96 192 720)
                elif [[ $amp == 5 ]]; then
                    command_args+=(96 192 336 720)
                elif [[ $amp == 10 ]]; then
                    command_args+=(96 336 720)
                fi

                mkdir -p "${save_dir}/"
                (
                    echo "Running command for $job_name"
                    bash ./scripts/ICML2024/ltf_srh_ax_amp/${dst}_script/${mdl}.sh "${command_args[@]}"
                ) 2>&1 | tee -a "${save_dir}/stdout.txt"
            done
        done


        # for nfp in 0.2; do
        #     for amp in 0.1 0.5 1 5 10 50; do
        #         job_name=${exp_name}_amp_$(printf "%g" "$amp")_ax_$(printf "%g" "$ax")_dst_${dst}_mdl_${mdl}_nfp_${nfp}
        #         save_dir="./${exp_name}/${job_name}"
        #         command_args=("${ax}" "${amp}" "${nfp}" "${gpu}" "${save_dir}")
        #         if [[ $amp == 0.1 ]]; then
        #             command_args+=(336 720)
        #         elif [[ $amp == 0.5 ]] || [[ $amp == 5 ]]; then
        #             command_args+=(192 336 720)
        #         elif [[ $amp == 1 ]]; then
        #             command_args+=(192 336)
        #         elif [[ $amp == 10 ]]; then
        #             command_args+=(192 720)
        #         else
        #             command_args+=(96 192 336 720)
        #         fi

        #         mkdir -p "${save_dir}/"
        #         (
        #             echo "Running command for $job_name"
        #             bash ./scripts/ICML2024/ltf_srh_ax_amp/${dst}_script/${mdl}.sh "${command_args[@]}"
        #         ) 2>&1 | tee -a "${save_dir}/stdout.txt"
        #     done
        # done
    done
done
