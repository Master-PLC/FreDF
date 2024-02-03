exp_name=test_code

gpu=0

dst=ECL
mdl=iTransformer

num_proc=2

job_name=imp_dst_${dst}_mdl_${mdl}
save_dir="./${exp_name}/${job_name}"
command_args=("${save_dir}" "${gpu}" "${num_proc}")
command_args+=(0.125)

mkdir -p "${save_dir}/"
(
    echo "Running command for $job_name"
    bash ./scripts/test_imp.sh "${command_args[@]}"
) 2>&1 | tee -a "${save_dir}/stdout.txt"