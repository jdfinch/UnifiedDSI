
mkdir subgit && cd subgit
git clone https://github.com/jdfinch/ezpyzy.git
cd ezpyzy && git checkout UnifiedDSI && cd ..

git clone https://github.com/jdfinch/language_model.git
cd language_model && git checkout UnifiedDSI && cd ..
cd ..

ln -s ../subgit/ezpyzy/ezpyzy $(realpath src/ezpyzy)
ln -s ../subgit/langauge_model/language_model $(realpath src/language_model)

# conda create -n UnifiedDSI python=3.11
# conda activate UnifiedDSI
# pip install -r requirements.txt

mkdir data
mkdir ex
mkdir results

source src/dsi/data/multiwoz24/download.sh
source src/dsi/data/sgd/download.sh
source src/dsi/data/dot/download.sh
