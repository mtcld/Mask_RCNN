## Things to do
- create json annotations
- Change class_id for load_mask


### Create json annotations
If annotations come as xml file do the following:
run copy_files to get all xml and images data in to 1 folder with train/test splited

`python copy_files -e jpg -i images -o img -r 0.8`

run convert_xml_json for each train and test folder to create json for its data

`python convert_xml_json.py -a annotations/ -i /images -o train.json`



### Note for convert_xml_json :
change this as label u want it to collect 
`if polygon.text in ['broken', 'crack', 'carck', 'broke', 'crackk']:`
