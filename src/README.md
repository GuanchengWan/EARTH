To run the EARTH model, simply execute the following command:

```bash
bash run.sh
```

This will train the EARTH model with default parameters on the US HHS epidemiological dataset.

For custom training, use:

```bash
python train.py --model earth_epi --dataset us_hhs --sim_mat us_hhs-adj --epochs 400 --lr 0.001 --horizon 1 --n_hidden 128
```
