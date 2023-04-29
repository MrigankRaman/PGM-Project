# Create Scene Graphs

Download the data by following instructions of [official NLVR2 website](https://lil.nlp.cornell.edu/nlvr/)

```
git clone https://github.com/microsoft/scene_graph_benchmark.git
```
Install all dependencies in the repo and then run the following command to generate scene graphs

```
python tools/demo/demo_image.py --config_file sgg_configs/vrd/R152FPN_vrd_reldn.yaml --img_file demo/1024px-Gen_Robert_E_Lee_on_Traveler_at_Gettysburg_Pa.jpg --save_file output/1024px-Gen_Robert_E_Lee_on_Traveler_at_Gettysburg_Pa.reldn_relation.jpg --visualize_relation MODEL.ROI_RELATION_HEAD.DETECTOR_PRE_CALCULATED False 
```



