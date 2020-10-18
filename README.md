# 2020-IA-bonnes-pratiques-notebook

The goal is to show how we can improve a very simple, notebook-based, project into a very readable, reusable and changeable one.  

Steps go as follows :
 - 0 : Initial project. Base code from [janakiev](https://janakiev.com/blog/keras-iris/) made a lot worse on purpose.
 - 1 : Improve the notebook itself : Add markdown, better and more pythonic code, etc.
 - 2 : Add some extra files : README.md and requirements.txt and use an isolated environment.
 - 3 : Separate notebooks as a DAG.
 - 4 : Externalise some of the code for better readability.
 - 5 : Unit test most of the externalised code.
 - 6 : Using Papermill for parametrized execution.

The overall final project architecture is a free interpretation of the [Cookie Cutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) project.

Extra interesting documentation on this matter : 
 - [Joel Grus - I don't like notebooks](https://www.youtube.com/watch?v=7jiPeIFXb6U)
 - [Dan Bader - Python Tricks](https://realpython.com/products/python-tricks-book/)
 - [Papermill](https://papermill.readthedocs.io/en/latest/)
 - [Jupyter Notebook Extensions](https://towardsdatascience.com/jupyter-notebook-extensions-517fa69d2231)
 - [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/)
 - [Working with Jupyter Notebooks in VSCode](https://code.visualstudio.com/docs/python/jupyter-support)
 - [TQDM](https://github.com/tqdm/tqdm)
 - [NbDime](https://nbdime.readthedocs.io/en/latest/)
 - [Ten Simple Rules for Reproducible Research in Jupyter Notebooks](https://arxiv.org/ftp/arxiv/papers/1810/1810.08055.pdf)
 - [Jupyter Notebook Best Practices](https://towardsdatascience.com/jupyter-notebook-best-practices-f430a6ba8c69)
 - [Working efficiently with JupyterLab Notebooks](https://florianwilhelm.info/2018/11/working_efficiently_with_jupyter_lab/)
 - [Bringing the best out of Jupyter Notebooks for Data Science](https://towardsdatascience.com/bringing-the-best-out-of-jupyter-notebooks-for-data-science-f0871519ca29)
 - [Jupyter Notebook Manifesto: Best practices that can improve the life of any developer using Jupyter notebooks](https://cloud.google.com/blog/products/ai-machine-learning/best-practices-that-can-improve-the-life-of-any-developer-using-jupyter-notebooks)
 - [Jupyter Lab: Evolution of the Jupyter Notebook](https://towardsdatascience.com/jupyter-lab-evolution-of-the-jupyter-notebook-5297cacde6b)
 - [PEP8](https://www.python.org/dev/peps/pep-0008/)