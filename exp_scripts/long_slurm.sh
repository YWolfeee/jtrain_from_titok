#!/bin/bash
 
 
# Usage:
#  1. Create one copy of this script per long running job. Use unique job name for each long running job. (#SBATCH --job-name).
#  2. Adjust other sbatch options below.
#  3. Check/Adjust TODOs.
#  4. Set MAX_JOB_COUNT and MAX_RETRIES to not very large numbers.
#  5. Monitor your job for any major reliability or performance issues.
#
#  Start a long running job:
#    $ sbatch <script>
#
#  Cancel long running job (See STATE_DIR variable below for <state dir>):
#    $ touch <state dir>/cancel
#
#  Continue a job that stopped after hitting some limit:
#    * If MAX_JOB_COUNT was reached, increase it.
#    * If MAX_RETRIES was reached, after fixing any bugs, increase it.
#    * If persistent failure was hit, fix the issue, increase MAX_RETRIES.
#    $ sbatch <script>
#
#  For any assistance needed in debugging job failures, ask for help in the slack support channel for the cluster where jobs
#  were scheduled.
 

#SBATCH --job-name=${SLURM_JOB_NAME}     # Specify job name, note that this is they primary "key" for the chained-jobs.
                                                        # Use different job names for multiple concurrent running long jobs.
 
 
#SBATCH --output=outputs/${SLURM_JOB_NAME}/logs/slurm_%j.out # Specify stdout file - %j will be substituted with job ID
 
 
#SBATCH --time=04:00:00             # Time limit
#SBATCH --account=dir_cosmos_misc
#SBATCH --partition=batch
#SBATCH --gpus-per-node=8
#SBATCH --no-requeue                # Control all retries/attempts explicitly
#SBATCH --dependency=singleton      # Singleton dependency - run one job at a time

# the main key of the run is ${SLURM_JOB_NAME}, which needs to be identical 
# from runs to runs

# The following arguments are passed to main.sh, which is seperate
config_name=$1      # name of the `yaml` to call, 'titok_l256_4096_12'
per_gpu_batch_size=$2 # 64
learning_rate=$3 # 2e-4
use_reconstruction_regularization=$4         # use_reconstruction_regularization True
use_annealing=$5    # False
is_increasing=$6    # False


echo "Running config: $config_name; batch_size: ${per_gpu_batch_size}; learning_rate: ${learning_rate}; user_reconstruction_regularization: ${use_reconstruction_regularization}; use_annealing: ${use_annealing}; is_increasing: ${is_increasing}."

output_root='results'

# Enable strict error handling to improve script reliability.
set -euo pipefail
 
 
# Log messages with timestamp
log_msg() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - [JobId=${SLURM_JOB_ID}] - $1"
}
 
 
# Send email notification to the user
send_mail() {
    local subject="$1"
    local message="$2"
    local recipient="${USER}@nvidia.com"
 
 
    log_msg "${subject}"
     
    # Send the email
    # TODO(vikasm): uncomment after getting mail server working in CS clusters.
    # echo "${message}" | mail -s "[JobName=${SLURM_JOB_NAME}][Cluster=${SLURM_CLUSTER_NAME}] $subject" "$recipient"
}
 
 
# Log the command that failed
error_handler() {
    local exit_status=$?
    local command="${BASH_COMMAND}"
    log_msg "Error: ${command} exited with status ${exit_status}"
    exit $exit_status
}
 
 
trap error_handler ERR
 
 
# sbatch script that will be used to launch the next job in the sequence
readonly SBATCH_SCRIPT=$(readlink -f "$0")
 
 
# State that is passed from one job to the next in the chain.
STATE_DIR="${output_root}/${SLURM_JOB_NAME}/state"
STATE_FILE="${STATE_DIR}/${SLURM_JOB_NAME}.json"
CANCEL_JOB_FILE="${STATE_DIR}/cancel"
EXP_RUN_NAME="${SLURM_JOB_NAME}"
 
 
mkdir -p ${STATE_DIR}
 
 
# This configuration controls how many times job will be rescheduled even after hitting a failure
# (independent of failure reason). Recommendation is to keep this small to prevent retrying
# persistent issues like bug in the application code, missing/broken dependency (model/dataset
# directory changed), disk quota exceeded, etc.
#
# TODO: Adjust this for failure tolerance.
readonly MAX_RETRIES=3 

 
# This configuration controls the maximum number of jobs that will be scheduled, independent of
# their completion state. This ensures that we don't end up scheduling a large number of broken jobs.
#
# TODO: This can be calculated as:
#    (Expected total runtime + buffer / partition runtime limit (typically 4h)) + MAX_RETRIES
#  eg: Expect synthetic data generation job to take 36 hrs to process all the samples and use
#      8 hrs buffer.
#     MAX_JOB_COUNT = (36+8)/4 + 5 = 16
readonly MAX_JOB_COUNT=20
 
 
STORED_MAX_JOB_COUNT=0
STORED_MAX_RETRIES=0
STORED_FAILURE_COUNT=0
STORED_NUM_JOBS_SCHEDULED=1
STORED_NUM_JOBS_COMPLETED=0
STORED_UPDATER_JOBID=""
 
 
PREV_JOB_SUCCEEDED=1
 
 
# Functions read_state/write_state are used to pass meta-data between jobs in the chain.
function read_state() {
    STORED_MAX_JOB_COUNT=$(jq -r '.max_job_count' ${STATE_FILE})
    STORED_MAX_RETRIES=$(jq -r '.max_retries' ${STATE_FILE})
    STORED_NUM_JOBS_SCHEDULED=$(jq -r '.num_jobs_scheduled' ${STATE_FILE})
    STORED_NUM_JOBS_COMPLETED=$(jq -r '.num_jobs_completed' ${STATE_FILE})
    STORED_FAILURE_COUNT=$(jq -r '.failure_count' ${STATE_FILE})
    STORED_UPDATER_JOBID=$(jq -r '.updater_job_id' ${STATE_FILE})
   
    log_msg "Read state:
      MAX_JOB_COUNT=${STORED_MAX_JOB_COUNT},
      MAX_RETRIES=${STORED_MAX_RETRIES},
      FAILURE_COUNT=${STORED_FAILURE_COUNT},
      NUM_JOBS_SCHEDULED=${STORED_NUM_JOBS_SCHEDULED},
      NUM_JOBS_COMPLETED=${STORED_NUM_JOBS_COMPLETED},
      UPDATER_JOBID=${STORED_UPDATER_JOBID}"
}
 
 
function update_state() {
    # Create a JSON object  
    state_json=$(jq -n \
            --argjson max_job_count "${STORED_MAX_JOB_COUNT}" \
            --argjson max_retries "${STORED_MAX_RETRIES}" \
            --argjson failure_count "${STORED_FAILURE_COUNT}" \
            --argjson num_jobs_scheduled "${STORED_NUM_JOBS_SCHEDULED}" \
            --argjson num_jobs_completed "${STORED_NUM_JOBS_COMPLETED}" \
            --arg updater_job_id "${SLURM_JOB_ID}" \
            '{
                    max_job_count: $max_job_count,
                    max_retries: $max_retries,
                    failure_count: $failure_count,
                    num_jobs_scheduled: $num_jobs_scheduled,
                    num_jobs_completed: $num_jobs_completed,
                    updater_job_id: $updater_job_id}')
 
 
    log_msg "Updated state: "
    echo "$state_json" 2>&1 | tee ${STATE_FILE}
}
 
 
function launch_more_jobs() {
    local num_jobs_to_launch=${PREV_JOB_SUCCEEDED}
 
 
    # Did MAX_JOB_COUNT or MAX_RETRIES confg change?
    if [ ${MAX_RETRIES} -gt ${STORED_MAX_RETRIES} ] || [ ${MAX_JOB_COUNT} -gt ${STORED_MAX_JOB_COUNT} ]
    then
    # start of pipeline, we want to schedule (MAX_RETRIES - STORED_MAX_RETRIES) number of jobs
    num_jobs_to_launch=$((MAX_RETRIES - STORED_FAILURE_COUNT - 1))
    if [ ${num_jobs_to_launch} -lt 1 ]
    then
        num_jobs_to_launch=1 # atleast launch one job until we have reached MAX_JOB_COUNT
    fi
 
 
    STORED_MAX_RETRIES=${MAX_RETRIES}
    STORED_MAX_JOB_COUNT=${MAX_JOB_COUNT}
    fi
     
    local num_remaining_jobs=$((MAX_JOB_COUNT - STORED_NUM_JOBS_SCHEDULED))
 
 
    if [ ${num_jobs_to_launch} -gt ${num_remaining_jobs} ]
    then
    num_jobs_to_launch=${num_remaining_jobs}
    fi
    log_msg "Launching ${num_jobs_to_launch} jobs..."
 
    command="sbatch --job-name=${SLURM_JOB_NAME} --output='${output_root}/${SLURM_JOB_NAME}/logs/slurm_%j.out' exp_scripts/long_slurm.sh $config_name $per_gpu_batch_size $learning_rate $use_reconstruction_regularization $use_annealing $is_increasing"
    echo "$command"
    for ((i = 1; i <= ${num_jobs_to_launch}; i++)); do
    log_msg "[JobId=${SLURM_JOB_ID}] Launching next job..."
    # sbatch --job-name=${SLURM_JOB_NAME} --output="${output_root}/${SLURM_JOB_NAME}/logs/slurm_%j.out" ${SBATCH_SCRIPT} "$@"
    eval "$command"
    let STORED_NUM_JOBS_SCHEDULED+=1
    done
}
 
 
# This function checks if there is more work left to do. Return zero value when there is more
# work to do. This function is used to determine whether we should enqueue more jobs or not. Not
# implementing this function isn't fatal, but it will be wasteful as this script will continue
# to enqueue jobs until we hit the max-retries or fail safe total job count threshold, which will
# be waste a lot of resources.
#
# TODO: implement it
function is_job_done() {
    local file_path="${output_root}/${EXP_RUN_NAME}/done.txt"
    if [[ -f "$file_path" ]]; then
        log_msg "Job is done (found done.txt)..."
        return 0 # done
    else
        log_msg "Job is not done (done.txt not found)..."
        return 1 # not done
    fi
}

 
# Check previous job run state to determine whether failure is transient/retriable (return 1)
# or fatal non-retriable (return 0).
#
# TODO: implement it
function is_fatal_non_retriable_error() {
    log_msg "Previous job failure is not fatal, we can continue (not implemented)..."
    return 1 # not fatal, can continue. return 0 when job cannot be continued
}
 
 
# This is the main function for the acutal work to be done in this job.
function do_actual_work() {
    # TODO: Replace below with your job commands here
    # sleep 30
    export ENROOT_DATA_PATH="/lustre/fsw/portfolios/dir/users/haotiany/workspaces"
    # enroot list -f
    # pwd
    enroot start --rw --mount /lustre/fsw/portfolios/dir/users/haotiany/joint_training/:/joint_training my_workspace \
        /bin/bash /joint_training/jtrain_from_titok/exp_scripts/main.sh $config_name $per_gpu_batch_size $learning_rate $use_reconstruction_regularization $use_annealing $is_increasing ${output_root} ${SLURM_JOB_NAME}
        # /bin/bash -c "source ~/.bashrc; pip show torchinfo; which accelerate; cd /joint_training/jtrain_from_titok; export PYTHONPATH='/joint_training/jtrain_from_titok'; \
        # accelerate launch \
        # --num_machines=1 --num_processes=4 --machine_rank=0 \
        # --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network \
        # scripts/train_titok.py config=configs/training/stage1/titok_s128_matryoshka_annealing.yaml \
        # experiment.project='${EXP_RUN_NAME}' \
        # experiment.name='${EXP_RUN_NAME}_run1' \
        # experiment.output_dir='${EXP_RUN_NAME}_run1' \
        # training.per_gpu_batch_size=32 
        # "
 
    # Simulate a coin toss: generate a random number
    toss=$((RANDOM % 10))
    if [ "$toss" -le 8 ]; then
    log_msg "Coin toss result: Success (toss=$toss)"
    else
    log_msg "Coin toss result: Failure (toss=$toss)"
    # simulate failure
    cat nonexistent_file.txt
    fi
}
 
 
# Should launch another job after this or not?
#   Reasons:
#     * Job is not explicitly cancelled.
#     * Work is not completed.
#     * We have not hit a fatal non-retriable error.
#     * Have not exhaused number of retries for failures.
#     * Have not reached the MAX_JOB_COUNT.
 
 
if [ -f ${CANCEL_JOB_FILE} ]
then
    log_msg "Explicit cancel requested, not going to continue."
    exit 0
fi
 
 
if [ -f ${STATE_FILE} ]
then
    read_state
   
    if is_job_done
    then
    send_mail "Job completed successfully!" "All work is completed!"
    exit 0
    fi
 
 
    if [ -e $STORED_UPDATER_JOBID ]
    then
    log_msg "Updater job id not found. Not going to continue...";
    send_mail "[FATAL] Job failed due to missing state!" "Please check the logs."
    exit 1
    fi
 
 
    # If previous job failed, bump the failure count.
    prev_job_state=$(sacct -Xj ${STORED_UPDATER_JOBID} --format=state --noheader | head -1 | xargs)
    if  [ $prev_job_state != "COMPLETED" ] && [ $prev_job_state != "TIMEOUT" ] 
    then
    let STORED_FAILURE_COUNT+=1
    PREV_JOB_SUCCEEDED=0
 
 
    if is_fatal_non_retriable_error
    then
        send_mail "[FATAL] Non-retriable error detected. Not going to continue!" "Please check the logs."
        exit 1
    fi
    fi
   
    # Stop if max-retries reached.
    if [ ${STORED_FAILURE_COUNT} -ge ${MAX_RETRIES} ]
    then
    send_mail "[FATAL] Retries (RetryCount=${MAX_RETRIES}) exhausted. Not going to continue!" "Please check the logs."
    exit 1 # Indicate that overall we couldn't complete all the work.
    fi
 
 
    let STORED_NUM_JOBS_COMPLETED+=1
     
    # Stop if max-retries reached.
    if [ ${MAX_JOB_COUNT} -le ${STORED_NUM_JOBS_COMPLETED} ]
    then
    send_mail "Completed configured max job count (${MAX_JOB_COUNT}) runs. Not going to continue!" "Increase MAX_JOB_COUNT if work is not done."
    fi
fi
 
 
# Launch more jobs before doing any work to ensure that we have a backup in case this job
# run into failure/time limit and we don't get a chance to enqueue next job.
launch_more_jobs
 
 
update_state
 
 
do_actual_work

EOF