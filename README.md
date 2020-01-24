# Increasing the robustness of DNNs against imagecorruptions by playing the Game of Noise

This repository contains trained model weights and evaluation code for the paper [Increasing the robustness of DNNs against image corruptions by playing the Game of Noise](https://arxiv.org/abs/2001.06057) by Evgenia Rusak*, Lukas Schott*, Roland Zimmermann*, Julian Bitterwolf, Oliver Bringmann, Matthias Bethge & Wieland Brendel.

Our core message is that a very simple approach -- data augmentation with Gaussian noise -- suffices to surpass almost all much more sophisticated state-of-art methods to increase robustness towards common corruptions. Going one step further, we learn the per-pixel distribution to sample noise from adversarially with a simple generative neural network which we call the Noise Generator. Training the Noise Generator and the classifier jointly further increases robustness.

![Example Figure](./Figures/Fig1.png)

## Evaluate models

Download and validate our models via:

```
python3 main.py --model_name ANT-SIN
```

The results are saved as txt files and displayed via print() statements directly. The ImageNet-C results are additionally saved as arrays.
