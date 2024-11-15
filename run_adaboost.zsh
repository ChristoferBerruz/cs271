# SOURCE_DIR - takes the directory where all the embedding datasets are stored
# RESULTS_DIR - takes the directory where the results will be stored

# check if source directory is set, if not set it to the default
if [ -z $SOURCE_DIR ]; then
    SOURCE_DIR=/home/cberruz/CS271/project_data/EmbeddedDatasets
fi
# check if results directory is set, if not set it to the default
if [ -z $RESULTS_DIR ]; then
    RESULTS_DIR=/home/cberruz/CS271/project_data/results
fi

subset=$1

# if subset is empty, only match .csv files
if [ -z $subset ]; then
    subset="*.csv"
fi
# if subset is not empty, match the subset
if [ ! -z $subset ]; then
    subset="*$subset*.csv"
fi

for file in $(find $SOURCE_DIR -name "$subset");
do
    python -m mlproject adaboost --ds=$file --save-dir=$RESULTS_DIR;
done