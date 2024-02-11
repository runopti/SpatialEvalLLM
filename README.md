# SpatialEvalLLM
Code for ["Evaluating Spatial Understanding of Large Language Models" TMLR 2024](https://arxiv.org/abs/2310.14540).

### Requirements
```
numpy
inflect (for programatically identifying correct indefinite articles for nouns)
networkx (for tree maps)
```

### Local map

- Square:
```
python sample_maps.py --size 3 --seed 3 --n_sample 10000 --maptype square \
--label_path labels/imagenetsimple.json --out_dir results_map_local
```

- Ring:
```
python sample_maps.py --size 8 --seed 8 --n_sample 10000 --maptype ring \
--label_path labels/imagenetsimple.json --out_dir results_map_local 
```

- Hexagon:
```
python sample_maps.py --size 1 --seed 1 --n_sample 5 --maptype hexagon \
--label_path labels/imagenetsimple.json --out_dir results_map_local
```

- Triangle:
```
python sample_maps.py --size 2 --seed 2 --n_sample 5 --maptype triangle \
--label_path labels/imagenetsimple.json --out_dir results_map_local 
```

- Rhombus
```
python sample_maps.py --size 3 --seed 3 --n_sample 10000 --maptype rhombus \
--label_path labels/imagenetsimple.json --out_dir results_map_local
```

- Rectangle:
```
python sample_maps.py --size 2 3 --seed 1 --n_sample 50 --maptype rectangle \
--label_path labels/imagenetsimple.json --out_dir results_sizeinference 
```

After sampling a map using one of the above commands, 
you can further filter the map to contain a specific number of steps by running:
```
python sample_from_jsonl.py \
--pred-file memory_local/type-square_size-3_seed-3_n-10000_label-imagenetsimple.jsonl \
--nstep 8 --nsampled 200
```

### Global map

- Square:
```
python square.py --seed 3 --size 3  --steps 8 --maptype square \
--label_path ./labels/imagenetsimple.json --n_sample 1000 --out_dir results_map_global
```

- Ring:
```
python ring.py --seed 12 --size 12  --steps 8 --maptype ring \
--label_path labels/imagenetsimple.json  --n_sample 1000 --out_dir results_map_global
```

- Tree:
```
python tree.py --size 9 --seed 9 --steps 4  --n_sample 100 \
--maptype tree --label_path labels/imagenetsimple.json \
--out_dir results_map_global --save_tree
```

