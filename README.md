# Create Scene Graphs

Download the data by following instructions of [official NLVR2 website](https://lil.nlp.cornell.edu/nlvr/) and then follow the following instructions to create the scene graphs

```
git clone https://github.com/microsoft/scene_graph_benchmark.git
```
Install all dependencies in the repo and then adjust the following command for your images to generate the scene graphs

```
python tools/demo/demo_image.py --config_file sgg_configs/vrd/R152FPN_vrd_reldn.yaml --img_file demo/1024px-Gen_Robert_E_Lee_on_Traveler_at_Gettysburg_Pa.jpg --save_file output/1024px-Gen_Robert_E_Lee_on_Traveler_at_Gettysburg_Pa.reldn_relation.jpg --visualize_relation MODEL.ROI_RELATION_HEAD.DETECTOR_PRE_CALCULATED False 
```

# Setting up the environment

    conda create -n SGNN python=3.10
    pip install torch==1.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.12.1+cu116.html
    pip install transformers
    


# Run the model

Once scene graphs are generated, change the path of the scene graphs and the data and run the following command to train and evaluate the model

```
python gcn.py
```

We release the pretrained features used by us and our final checkpoint [here](https://drive.google.com/drive/folders/18dZGPH1G1RoJ3tSZ2FEYyYABTm__85hr?usp=sharing)



