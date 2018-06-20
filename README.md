# DeepPano

## Necessary Setup
make following directories
- `DeepPano/data/`
- `DeepPano/data/metadata/` 
- `DeepPano/data/rawdata/`
- `DeepPano/data/rawdata/panoImg`: original panorama images
- `DeepPano/data/rawdata/psdFile`: annotation files
- `DeepPano/data/rawdata/xmlFile`: PanoSeg or box information
- `DeepPano/result/`
- `DeepPano/result/checkpoint`: for postprocess


## To Generate FirstFile.csv
### `python3 firstfilegen.py semi`
it iterates `DeepPano/data/rawdata/panoImg` files to generate FirstFile for all panorama images

and generate `DeepPano/data/metadata/SemiSet.csv`
### `python3 firstfilegen.py not`
it iterates `DeepPano/data/rawdata/psdFile` files to generate FirstFile for all annotated panorama images

and generate `DeepPano/data/metadata/DataSet.csv`


or you can use your own .csv file that has `Image.Title`, `Pano.File`, `Xml.File`, `Annot.File`, `Train.Val` columns


## To Generate Image Patches And According .csv
### `python3 manage.py genData 10 (FirstFile.csv route)`
This generates image patches in `DeepPano/data/metadata/(firstfile name)/`

and metadata of `DeepPano/data/metadata/(firstfile name)-10.csv`

To use this dataset for training, you need to make `DeepPano/data/StatDataset.csv`

### `python3 manage.py genStat (SecondFile.csv route)`
It generates pano/box mean/std stat for the dataset


## To Generate Result
### Necessary Setup
put checkpoint under `DeepPano/result/checkpoint/(directory name)/checkpoint/`

and config file(.json) used to train that checkpoint at `DeepPano/result/checkpoint/(directory name)/`

SecondFile.csv and its Dataset, and its panoImgs should also be prepared

### `python3 postprocess.py -1 (SecondFile.csv route)`
This iterates all folders in `DeepPano/result/checkpoint/` and generate result for all image patches in SecondFile.csv
### `python3 postprocess.py (checkpoint directory route) (SecondFile.csv route)`
This iterates all checkpoints in checkpoint directory and generate result for all image patches in SecondFile.csv
