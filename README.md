# Machine Learning, AI, LLM study

## Math Basics
+ **Statistics Fundamentals**
    + Histogram, mean, variance, standard deviations, and [more](./https://www.youtube.com/watch?v=qBigTkBLU6g&list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9)
+ **Linear Algebra Fundamentals**
    + [Essence of linear algebra](https://www.youtube.com/watch?v=fNk_zzaMoSs&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
    + [Art of Linear Algebra](https://github.com/kenjihiranabe/The-Art-of-Linear-Algebra/blob/main/The-Art-of-Linear-Algebra.pdf)

## Topics
+ **Machine Learning wiith Scikit-Learn**
    + [Data preprocessing](./DataPreprocessing.md)
    + Classifiation: [KNN](./learn-knn.py)
        + [K-nearest neighbors, Clearly Explained](https://www.youtube.com/watch?v=HVXime0nQeI&t=259s)
    + Classifiation: [Naive Bayes](./learn-naivebayes.py)
        + [Naive Bayes, Clearly Explained!!!](https://www.youtube.com/watch?v=O2L2Uv9pdDA)
        + [Gaussian Naive Bayes, Clearly Explained!!!](https://www.youtube.com/watch?v=H3EjCKtlVog)
    + Classifiation: [Decision Tree](./learn-decisiontree.py)
        + [Entropy Clearly Explained!!!](https://www.youtube.com/watch?v=YtebGVx-Fxw&t=882s)
        + [Decision and Classification Trees, Clearly Explained!!!](https://www.youtube.com/watch?v=_L39rN6gz7Y)
    + Clustering: [K-means](./learn-kmean.py)
        + [K-means clustering](https://www.youtube.com/watch?v=4b5d3muPQmA)
    + Regression: [Gradient Descend](./learn-regression-gradientdescend.py)
        + [Linear Regression, Clearly Explained!!!](https://www.youtube.com/watch?v=7ArmBVF2dCs)
        + [Gradient Descent, Step-by-Step](https://www.youtube.com/watch?v=sDv4f4s2SB8&t=1063s)
    + Dimension Reduction: Principal Component Analysis [PCA](./learn-pca-dimemreduction.py)
        + [PCA main ideas in only 5 minutes!!!](https://www.youtube.com/watch?v=HMOI_lkzW08)
    + AutoML 
        + AutoML automates data preprocessing, feature selection, model selection, hyperparameter tuning, and model evaluation.
        + [Google AutoML](https://cloud.google.com/vertex-ai/docs/beginner/beginners-guide?_gl=1*sxk3eq*_up*MQ..&gclid=CjwKCAjwjeuyBhBuEiwAJ3vuoVt6pz25to08norJPNRM8TU3zxRMoX5ZzB2GZsDb9gEj2STOrOnyDRoCZS8QAvD_BwE&gclsrc=aw.ds)
        + [auto-sklearn](https://automl.github.io/auto-sklearn/master/)
    + Scikit Learn features
        + [Pipeline](./scikit-pipeline.py)
            + The scikit-learn pipeline streamlines and automates the machine learning workflow. It combines multiple steps of data preprocessing and modeling into a single object.
        + [Ensemble Learn](./EnsembledLearn.md)
            + Combining multiple machine learning models to improve overall performance.
                + [Majority Vote](./ensembled-learn.py)
                + Bagging (Bootstrap Aggregating)
                + Boosting
                + Stacking
        + [Model Evaluation and Hyperparameter Tuning](./Evaluation.md)
+ **Artifical Neural Network (ANN) Fundamentals**
    + [Neural network](https://www.youtube.com/watch?v=zxagGtF9MeU&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1)
    + Gradient Descent and Backpropagation
    + [Tensors and PyTorch](https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4)
+ **Deep Learning** [DL](./DeepLearning.md)
+ **Reinforcement Learning** [RL](./ReinforcementLearning.md)
+ **DL-RNN-Self Attention-Large Language Models** [LLM](https://www.youtube.com/playlist?list=PLz-ep5RbHosU2hnz5ejezwaYpdMutMVB0)
    + Prompt engineering (e.g. Microsoft [Lida](https://github.com/microsoft/lida) [explained](./MicrosoftLida.md))
        + Prompt engineering [guide](https://www.promptingguide.ai/)
    + OpenAI API, Huggingface, LangChain, Ollama
        + Use LangChain: When building complex applications requiring multiple steps, integrating various models and tools, and needing to manage custom workflows.
        + Use Hugging Face: For accessing pre-trained models, training and fine-tuning models, and developing straightforward NLP applications with ease.
        + Use Ollama: Provides a seamless way to run open-source LLMs locally. Use Ollama in [Google Codelab](https://blog.gopenai.com/run-ollama-llama3-llm-on-google-colab-9b56b7254be9)
    + RAG
        + [How RAG works](./RAG.md)
        + [Fine-tuning RAG](./RAGTuning.md)
    + Fine-tuning a LLM
        + [Methods of fine-tuning](./FineTuneLLM.md)
        + PEFT Tuning: [Unsloth fine-tuning jupyter books](https://github.com/unslothai/unsloth?tab=readme-ov-file)
            + [Example](https://www.youtube.com/watch?v=WxQbWTRNTxY)
    + [Build your own LLM](https://towardsdatascience.com/a-complete-guide-to-write-your-own-transformers-29e23f371ddd)
    + [LLM Applications](./LLMApplications.md)
+ **Using Colab** (Cloud virtual machine + GPU + Jupyterbook) for ML and AI
    + [Intro](https://www.youtube.com/watch?v=inN8seMm7UI)

## References:
+ Book: https://www.google.com/books/edition/Data_Science_Algorithms_in_a_Week/Pel1DwAAQBAJ?hl=en&gbpv=1&printsec=frontcover
+ Book (Scikit+PyTorch): https://www.packtpub.com/product/machine-learning-with-pytorch-and-scikit-learn/9781801819312
+ Book (Scikit+TensorFlow): https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646
+ Book (TensorFlow+Keras): https://download.bibis.ir/Books/Artificial-Intelligence/Deep-Learning/2021/Deep%20Learning%20with%20Python%20by%20Franc%CC%A7ois%20Chollet_bibis.ir.pdf
+ Scikit Learn <img src="./scikit-learn_map.png" alt="plugin" style="zoom: 50%;" />
+ Types of machine learning [explained](./MachineLearning.md)
+ Tensorflow vs.Keras [compared](./TensorFlowKeras.md)