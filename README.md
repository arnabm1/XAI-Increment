Modular Implementation of the **EUSIPCO 2023 paper "[Harnessing the Power of Explanations for Incremental Training: A LIME-Based Approach](https://arxiv.org/abs/2211.01413)"** 

**Structure of the Code**
- **IL_train.py:** 	Main file to perform training with incremental (IL) sessions and ewc (#sessions are user-defined)
- **data_prepare.ipynb:** Creates numpy arrays based on the user defined splits for the Google Speech Commands dataset
- **ewc.py:** Standalone implementation of weighted loss (wl) and ewc based incremental training (fixed for six IL sessions)
- **inc.py:** Standalone implementation of conventional crossentroopy based incremental training (fixed for six IL sessions)
- **run_IL_train_ewc.sh:** 	Shell script for running the IL_train file with 'ewc' mode for multiple runs with different seeds (wl coupled with ewc based IL)
- **run_IL_train_trad.sh:** Shell script for running the IL_train file with 'trad' mode for multiple runs with different seeds (only wl based IL)
- **run_IL_train_wl.sh:** Shell script for running the IL_train file with 'wl' mode for multiple runs with different seeds (conventional crossentropy based IL)
- **test.ipynb:** Testing notebook to validate the execution incremental training
- **utils.py:** Helpfer functions for the implementation
- **wl.py:** Standalone implementation of weighted loss (wl) based incremental training (fixed for six IL sessions)

**Requirements**
- tensorflow
- numpy
- scikit-image
- scikit-learn
- 
