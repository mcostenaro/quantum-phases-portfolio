# Third-party notices

The project license is in `LICENSE` and retains the research author's original
copyright notice. It covers project-owned code and documentation, not external
datasets or dependencies.

`C_GAN/src/gan_ydata_like.py` identifies its Generator and Discriminator as
PyTorch versions of the YData Synthetic VanillaGAN architecture. This
attribution is retained. YData Synthetic's MIT notice is included verbatim in
`third_party/YDATA_LICENSE.txt`; upstream reference:
https://github.com/Data-Centric-AI-Community/fg-data-synthetic (formerly
https://github.com/ydataai/ydata-synthetic). Its license was checked for this
publication. This notice does not imply upstream endorsement.

NumPy, pandas, SciPy, scikit-learn, PyTorch, Matplotlib and pytest are installed
as dependencies and retain their respective upstream licenses; their source
trees are not vendored here. The source research implementations of the DDPM,
controls, split and classifiers retain the project MIT notice. Conceptual
references to GAN/DDPM papers are in `docs/DATA.md`.

External physical datasets, notebooks and papers are not redistributed and
are not relicensed by this repository. Attribution and the known provenance
limits are documented in `docs/DATA.md`.
