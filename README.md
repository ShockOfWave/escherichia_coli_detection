# Electrochemical platform based on eGaIn with hydrogel for detecting Escherichia coli bacteria using machine learning methods
## Project overview

Escherichia coli (E. coli) is a common pathogen that can cause diarrhea in both humans and animals. It forms a biofilm on various surfaces, leading to food contamination. The viability of the bacteria is crucial for them to cause harm by entering the gastrointestinal tract in sufficient quantities. Recent research has shown that E. coli is the pathogen responsible for the highest number of deaths among individuals with advanced antimicrobial resistance, particularly in cases of bloodstream infections (BSI). Approximately a quarter of the 4 million deaths related to antimicrobial resistance were due to BSI caused by E. coli. Therefore, it is crucial to have a fast, accurate, and effective method for detecting E. coli bacteria.

We present an electrochemical platform designed to greatly reduce time-consumption of Escherichia coli bacteria detection from a commonly used 24-48-hour procedure to 30 minutes detection range. The presented approach is based on a complex system which includes gallium-indium alloy to provide conductivity and a hydrogel system to preserve bacteria during the analysis. The performed analysis provides the ability to work with extremely fragile bacteria species that are sensitive to environmental conditions, and prolong their lifetime cycle till the analysis finishes.
Furthermore, the work is dedicated to accurate and fast detection of Escherichia coli bacteria in different environments with the supply of machine learning methods. Electrochemical data obtained during the analysis is processed via multilayer perceptron model to identify i.e. predict bacterial concentration in the samples. The performed approach provides the effectiveness of bacteria identification in the range of $10^2$ to $10^9$ colony forming units per ml with the impressive average accuracy of 98\%. 
The proposed bioelectrochemical system combined with machine learning model is prospective to be widely used in various domains including food production, agriculture, biomedicine, and environmental sciences as the rapid and facile bacterial identification is highly requested.

## Code usage

### Install packages with [pip](https://pypi.org/project/pip/) or [conda](https://conda.io)

```bash
pip install -r requirements.txt
```

### Train models

To train [catboost](https://catboost.ai) model use

```bash
python -m src.models.train_catboost
```

To train multilayer perceptron model use

```bash
python -m src.models.train_mlp
```

To train random forest model use

```bash
python -m src.models.train_random_forest
```

### Run trained models example

Example in notebook folder