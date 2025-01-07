# ViG-Bias: Visually Grounded Bias Discovery and Mitigation

See this [link](https://arxiv.org/abs/2407.01996) for more detailed results on bias discovery and mitigation.

## Abstract

The proliferation of machine learning models in critical decision-making processes has underscored the need for bias discovery and mitigation strategies. Identifying the reasons behind a biased system is not straightforward, since in many occasions they are associated with hidden spurious correlations which are not easy to spot. Standard approaches rely on bias audits performed by analyzing model performance in predefined subgroups of data samples, usually characterized by common attributes like gender or ethnicity when it comes to people, or other specific attributes defining semantically coherent groups of images. However, it is not always possible to know a priori the specific attributes defining the failure modes of visual recognition systems. Recent approaches propose to discover these groups by leveraging large vision language models, which enable the extraction of cross-modal embeddings and the generation of textual descriptions to characterize the subgroups where a certain model is underperforming. In this work, we argue that incorporating visual explanations (e.g. heatmaps generated via GradCAM or other approaches) can boost the performance of such bias discovery and mitigation frameworks. To this end, we introduce Visually Grounded Bias Discovery and Mitigation (ViG-Bias), a simple yet effective technique which can be integrated to a variety of existing frameworks to improve both discovery and mitigation performance. Our comprehensive evaluation shows that incorporating visual explanations enhances existing techniques like DOMINO, FACTS and Bias-to-Text, across several challenging datasets, including CelebA, Waterbirds, and NICO++.

## Installation

For installation, please follow the instructions provided in the FACTS and Bias-to-Text repositories:

- [FACTS Installation Guide](https://github.com/yvsriram/FACTS?tab=readme-ov-file#installation-instructions)
- [Bias-to-Text Installation Guide](https://github.com/alinlab/b2t?tab=readme-ov-file#installation)

After following the above installation guides, install `grad-cam` with pip:

```bash
pip install grad-cam
```

## Usage

To run bias discovery for Bias-to-Text with ViG-Bias, use the following command:

```bash
python b2t.py \
    --dataset waterbirds \
    --model best_model_Waterbirds_erm.pth \
    --apply_masks \
    --mask_threshold 0.7
```

To run bias discovery for FACTS with ViG-Bias, use the following command:

```bash
python slice.py \
    --outputs_file outputs/waterbirds/seed_0/all_outputs.npy \
    --stopping_time 90 \
    --dataset_name waterbirds \
    --mask_threshold 0.7 \
    --apply_masks \
    --classification_model_path best_model_Waterbirds_erm.pth \
```

## Note

We have modified the Bias-to-Text and FACTS codebases to include visual grounding techniques. This project uses code from the following repositories:

- [FACTS](https://github.com/yvsriram/FACTS)
- [Bias-to-Text](https://github.com/alinlab/b2t)
