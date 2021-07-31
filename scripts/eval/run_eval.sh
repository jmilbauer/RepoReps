embedding_path=$1
tag_data=$2
lib_data=$3
rec_data=$4

python eval_tags.py ${tag_data} ${embedding_path}
python eval_libraroes.py ${lib_data} ${embedding_path}
python eval_recs.py ${rec_data} ${embedding_path}

