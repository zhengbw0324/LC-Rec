model=$1

set -x

for step in `ls ${model} | grep checkpoint | awk -F'-' '{ print $2 }'`
do
mkdir ${model}/tmp-checkpoint-${step}
mkdir ${model}/final-checkpoint-${step}
python ./zero_to_fp32.py ${model}/checkpoint-${step}/ ${model}/tmp-checkpoint-${step}/pytorch_model.bin
cp ${model}/*.json ${model}/tmp-checkpoint-${step}
python ./convert.py -s ${model}/tmp-checkpoint-${step} -t ${model}/final-checkpoint-${step}
cp ${model}/checkpoint-${step}/*.json ${model}/final-checkpoint-${step}
cp ${model}/*.json ${model}/final-checkpoint-${step}
cp ${model}/tokenizer* ${model}/final-checkpoint-${step}
cp ${model}/train* ${model}/final-checkpoint-${step}
#rm -rf ${model}/tmp-checkpoint-${step} ${model}/checkpoint-${step} ${model}/global_step${step}
#mv ${model}/final-checkpoint-${step} ${model}/checkpoint-${step}
done