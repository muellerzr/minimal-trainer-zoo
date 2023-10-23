# A loop which iterates over all *.py files in the current directory
# times their execution, and stores/appends it to `times.txt`

# Ideally all scripts (post model download, etc) should complete in < 10 min
# on a single T4

for file in *.py; do
    echo $file >> times.txt
    (time python $file) 2>&1 | grep "real" >> times.txt
done


