# gdpnum

[![CI](https://github.com/Felipe-Gomez/gdp-numeric/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Felipe-Gomez/gdp-numeric/actions/workflows/ci.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2503.10945-b31b1b.svg)](https://arxiv.org/abs/2503.10945)

Library for numerically computing the correct, non-asymptotic Gaussian Differential Privacy (GDP) guarantees for DP-SGD or similar privacy-preserving algorithms.
Correct, non-asymptotic GDP accurately and faithfully represents the privacy guarantees of many practical algorithms with just a single _mu_ parameter, enabling correct comparisons (unlike fixed epsilon/delta), and accurate conversions to interpretable notions of risk. See Gomez et al. (2026)[^1].

[^1]: [Gaussian DP for Reporting Differential Privacy Guarantees in Machine Learning](https://arxiv.org/abs/2503.10945). IEEE SatML 2026.

If you make use of the library or methods, please cite:
```bibtex
@inproceedings{gomez2026gaussian,
  title={Gaussian DP for Reporting Differential Privacy Guarantees in Machine Learning},
  author = {Gomez, Juan Felipe and Kulynych, Bogdan and Kaissis, Georgios and du Pin Calmon, Flavio and Hayes, Jamie and Balle, Borja and Honkela, Antti},
  booktitle={2026 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML)},
  year={2026},
  organization={IEEE}
}
```

## Quickstart

You can install the library with pip:
```
pip install gdpnum
```


### DP-SGD
To analyze DP-SGD, you can use:

```python
import gdpnum

mu, regret = gdpnum.dpsgd.get_mu_and_regret_for_dpsgd(
    noise_multiplier=9.4,
    sample_rate=0.328,
    num_steps=2000
)
# (1.5685621993129137, 0.0010208130697719753)
```

We get the numerically computed GDP mu parameter, and regret which
shows the goodness-of-fit of the GDP.

The library also includes an [Opacus-compatible](https://opacus.ai/api/accounting/iaccountant.html) accountant interface:
```python
import gdpnum

acct = gdpnum.dpsgd.CTDAccountant()
acct.step(noise_multiplier=9.4, sample_rate=0.328)
acct.get_mu_and_regret()
```

### General mechanisms
For general mechanisms, the library relies on the privacy loss distribution
objects from the `dp_accounting` package:

```python
import gdpnum

pld = ...

converter = gdpnum.PLDConverter(pld)
mu, regret = converter.get_mu_and_regret()
```

See an example for the US Census TopDown mechanism in the notebooks folder.

To get the tabulation of the trade-off curve function at recommended values of alpha from the paper,
simply run:
```
converter.get_beta()
```
