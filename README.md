# ELE489-Homework-1

This project was completed as part of the **Fundamentals of Machine Learning** course. I implemented the k-Nearest Neighbors (k-NN) algorithm from scratch and tested its performance on the **Wine Dataset** from the UCI Machine Learning Repository.

## Files

- `knn.py`: The implementation of the k-NN algorithm from scratch, supporting both **Euclidean** and **Manhattan** distances.
- `analysis.ipynb`: Jupyter Notebook where I explain each step of the project and visualize the results.
- `Homework1.py`: Python script version of the notebook.
- `wine.data`: The Wine dataset used for classification.

---

## Project Overview

- The **Wine dataset** contains 178 samples, each with 13 features and a class label (1, 2, or 3).
- The data was pre-processed using **scaling**, and I split the data into **training** (80%) and **test** (20%) sets.
- I implemented the **k-NN algorithm** from scratch, using a range of **k** values (1 to 29, odd numbers only) and compared the performance using **Euclidean** and **Manhattan** distance metrics.
- **Confusion matrices** and **classification reports** were generated to evaluate the model performance.

---

Open the `analysis.ipynb` notebook in **Jupyter** and run each cell step-by-step to see the results.



## Notes

This project helped me understand the workings of the **k-NN algorithm** and how to tune its parameters for optimal performance. The **accuracy vs. k** plot and confusion matrices were particularly useful for analyzing the model’s behavior with different values of **k** and distance metrics.

