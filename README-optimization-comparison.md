# DistilBERT Hyperparameter Optimization: PSO vs GA vs Bayesian

Below is a clear, direct comparison of PSO, Genetic Algorithm, and Bayesian Optimization for tuning DistilBERT hyperparameters (learning rate, batch size, dropout, frozen layers) in hate-speech detection — focused on which performs better and why.

## ✅ Which Optimization Algorithm Performs Better for DistilBERT?

- **Best overall**: Bayesian Optimization
- **Second**: Genetic Algorithm (GA)
- **Third**: PSO

But the best choice depends on: search space type, budget, and interactions between hyperparameters.

## ✅ Why Bayesian Optimization Performs Best (Usually)

Bayesian Optimization (BO) uses a probabilistic model (surrogate model, like Gaussian Process / TPE) to guess which hyperparameters will be best next. It learns from previous trials, so with limited GPU resources, it finds good values faster.

### Why BO works especially well for DistilBERT
- **Hyperparameters influence each other strongly**
  - LR interacts with frozen layers
  - Batch size interacts with learning rate
  - Dropout affects convergence stability
  - BO models these interactions mathematically, while GA and PSO search blindly.
- **Expensive evaluations**
  - Training DistilBERT even for few epochs costs time.
  - BO minimizes the number of evaluations by exploring smarter.
- **Mixed search space (continuous + discrete + categorical)**
  - LR → continuous
  - Dropout → continuous
  - Batch size → discrete
  - Frozen layers → integer categorical
  - BO handles this naturally.

### Typical results
- Faster convergence
- Best accuracy/F1 with fewer runs
- Most stable model

## ✅ How Genetic Algorithm Performs

GA is good when the search space is large and rugged with many local optima — which is somewhat true for BERT models.

### Strengths
- Works well with mixed parameter types
- Useful when interactions are nonlinear
- Good at exploration → can find global optima

### Weaknesses
- Needs many generations → expensive
- Can be slow to converge
- Random mutation reduces stability

### When GA beats BO
- When the search space is huge
- When evaluation noise is high
- When the model is not smooth (loss jumps)

For DistilBERT: GA is good, but not as efficient as BO.

## ✅ How PSO Performs (and why it may perform worse)

PSO is excellent for continuous optimization with smooth search landscapes. But DistilBERT hyperparameters do not form a smooth landscape.

### Why PSO struggles here
- Batch size and frozen layers are discrete
  - PSO relies on moving "particles" smoothly → not ideal.
- Model training is noisy
  - A particle might move toward a region that seemed good by chance.
- Strong parameter interactions hurt PSO
  - LR + batch size + dropout affect each other non-linearly.
- Premature convergence
  - Swarm often collapses into a bad region early.

### When PSO works well
- Pure continuous parameters
- Low-dimensional search
- Smooth objective

DistilBERT tuning does not satisfy these.

## ✅ Final Ranking Summary

| Algorithm             | Performance | Why                                                |
|-----------------------|-------------|----------------------------------------------------|
| Bayesian Optimization | 🥇 Best     | Efficient, handles mixed spaces, learns interactions |
| Genetic Algorithm     | 🥈 Good     | Strong exploration, works with discrete params     |
| PSO                   | 🥉 Weakest  | Struggles with discrete params + noisy objective   |

## 📌 Which one you should use?

If you want:
- **Best performance + fewer GPU hours** → Use Bayesian Optimization
  - Finds LR, dropout, frozen layers efficiently
  - Works great with DistilBERT
  - Most research papers favor BO for NLP hyperparameter tuning
- **To compare evolutionary methods** → Include GA
  - Good benchmark, actually competitive
- **Include PSO only for scientific comparison**
  - It is common in papers, but expect worse results

## ✅ If you want, I can generate:

- ✔ Experiment design
- ✔ Table of hyperparameter search spaces
- ✔ Pseudocode or Python code for BO / GA / PSO
- ✔ Expected results & discussion
- ✔ A comparison slide for your presentation

Just tell me!
