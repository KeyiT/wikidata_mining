# Classification in Wikidata

This project tries to mine wikidata for general purposes.

The demo built a RF classifier to determine the most probable manner of death base on
person information queried from Wikidata database.

## How to run

The project process data in json format.

**To preprocess the raw json data**, initialize an instance of `Preprocessor` and pass
the raw json data file path to the initializer:

`prepro = Preprocessor("/path/to/json/data/file")`

Create lists of categorical properties `cat_prop_list` (allowed to be multi-valued), numerical properties
`num_prop_list` (must be single-valued), and specify the label property name `label_prop`.

Fill the missing data:

`prepro.fill_missing_data(cat_prop_list, num_feat_list, label_prop)`

Encode categorical properties as binary features using the one-hot scheme, and combine the categorical
features, numerical features and labels into a list in json format. The first list element stores the
metadata. Write the preprocessed data into file:

`prepro.assemble_features_labels_to_json("/path/to/destination/json/data/location")`

**To train a random forest classifier and test the performance**, type following commands in the project
root directory:

`$ python build_random_forest_classifier.py --src_json_data /preprocessed/json/data/path --train_perform train/performance/dest/data/path(optional) --test_perform test/performance/dest/data/path(optional) --model_bin model/saving/data/path(optional)`

## Author

* Keyi Tang ([@keyit](kytangls92@gmail.com))

## Licence

Copyright © 2017 KeyiT
See [License](https://github.com/KeyiT/wikidata_mining/blob/master/LICENSE) for licensing information



