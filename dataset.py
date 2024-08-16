from datasets import load_dataset


data_dir = "/home/xs439-rudagg/stabledifusion/icons"


dataset = load_dataset("imagefolder", data_dir=data_dir)

print(dataset)
print(dataset.column_names)


