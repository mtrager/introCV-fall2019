Special Topics in Data Science, DS-GA 3001.005/.006

### Logistics

* DS-GA 3001.005 (Lecture)  
Tuesdays 2pm-3:40pm 60FA room 110 

* DS-GA 3001.006 (Lab)  
Tuesdays 3:50pm-4:40pm 60FA room 110 

* Office hours: Thursdays 2:00pm-4:00pm 60FA room 606


### Instructors

Jean Ponce (jean.ponce@inria.fr)  
Matthew Trager (matthew.trager@cims.nyu.edu)

TAs:  
Jiachen Zhu (jiachen.zhu@nyu.edu)  
Sahar Siddiqui (ss12414@nyu.edu)


### Grading


Four/five programming assignments (60% of the grade) + final project (40% of the
grade). Assignments should be submitted using the NYU class site.

* **Excercise 1** on camera calibration ([zip file](/homeworks/homework1.zip)).
Due on October 1st.
* **Excercise 2** on Canny edge detector ([zip file](/homeworks/homework2.zip)).
Due on October 22nd.
* **Excercise 3** on mean shift ([zip file](/homeworks/homework3.zip)).
Due on November 12th.
* **Final project:** list of suggested papers is available [here](https://docs.google.com/document/d/15wjCUedE69u1c5ijW3S407oxISkLNlnHvB8ztOSvUUg/edit?usp=sharing). Send an email to Matthew to validate a project.

<!-- 
__Assignments:__ There will be four/five programming assignments representing
60% of the grade. The supporting materials for the programming assignments
projects will be in Python.

__Final project:__ The final project will represent 40% of the grade. Each
project is based on a paper and a list of suggested papers is available here.
Feel free to ask for papers on a topic that you are interested in or propose a
paper. You are expected to understand and present the paper, but also to offer
some added value, such as experiments of your own, new interesting tests with
available code, or comparison with other relevant works. This will have to be
adapted depending on the paper. You will have to present your project (10
minutes + questions) and return a summary (2 pages max) of the essential points
that should be readable (and useful) for the other students in the class. -->

__Collaboration policy:__ You can discuss the assignments and final projects
with other students in the class. Discussions are encouraged and are an
essential component of the academic environment. However, each student has to
work out their assignment alone (including any coding, experiments, and
derivations) and submit their own report/notebook.


### Syllabus

* __Part I: Low level Computer Vision__  
  * Filters, edge detection, visual features.
  * Radiometry, shading and color.
* __Part II: 3D reconstruction__  
  * Camera models, one-view geometry.
  * Multi-view geometry, stereo, SFM.
* __Part III: Recognition__  
  * CNNs for object detection and semantic segmentation.


_References:_  
* D.A. Forsyth and J. Ponce, "Computer Vision: A Modern Approach", second edition, Pearson, 2011.
* R. Szeliski, "Computer Vision: Algorithms and Applications". ([PDF](http://szeliski.org/Book/))






### Lectures

| Week | Date | Topic                                    | Instructor    | Slides |
| -----|------|----------------------------------------------|--------|
| 1    | 9/3  | Course overview, image formation        |  JP   |    [Slides](/lectures/lect1.pdf)    |
| 2    | 9/10 | Camera geometry and calibration I       |  JP   |    [Slides](/lectures/lect2.pdf)    |
| 3    | 9/17 | Camera geometry and calibration II      |  JP   |    [Slides](/lectures/lect3.pdf)    |
| 4    | 9/24 | Linear and nonlinear filters, edge detection |  MT  | [Slides](/lectures/lect4.pdf)    |
| 5    | 10/1 | Local image features (Harris, SIFT), robust estimation (RANSAC, Hough transform) |  MT   | [Slides](/lectures/lect5.pdf)       |
| 6    | 10/8 | Radiometry and color  |  JP  | [Slides](/lectures/lect6.pdf)    |
|     | 10/15 | No class  |   |    |
| 7    | 10/22 | Texture and image segmentation  |  MT  |  [Slides](/lectures/lect7.pdf)  |
| 8    | 10/29 | Multi-view reconstruction  |  MT  |    |



### Acknowledgements

Much of the material for this course relies on the Computer Vision course given at [ENS Paris](http://imagine.enpc.fr/~aubrym/lectures/introvis19/) by Mathieu Aubry, Karteek Alahari, Ivan Laptev, and Josef Sivic. Many of the slides are taken from [James Hays](https://www.cc.gatech.edu/~hays/compvision/), [Svetlana Lazebnik](http://slazebni.cs.illinois.edu/spring18/), and [Derek Hoeim](https://courses.engr.illinois.edu/cs543/sp2017/). 

<!-- 
### Detailed Syllabus 

*  Introduction: the Curse of Dimensionality

* Part I: Geometry of Data
  * Euclidean Geometry: transportation metrics, CNNs , scattering. 
  * Non-Euclidean Geometry: Graph Neural Networks. 
  * Unsupervised Learning under Geometric Priors (Implicit vs explicit models, microcanonical, transportation metrics).
  * Applications and Open Problems: adversarial examples, graph inference, inverse problems.

* Part II: Geometry of Optimization and Generalization
  * Stochastic Optimization (Robbins & Munro, Convergence of SGD) 
  * Stochastic Differential Equations (Fokker-Plank, Gradient Flow, Langevin Dynamics, links with SGD; open problems) 
  * Dynamics of Neural Network Optimization (Mean Field Models using Optimal Transport, Kernel Methods) 
  * Landscape of Deep Learning Optimization (Tensor/Matrix factorization, Deep Nets; open problems). 
  * Generalization in Deep Learning. 
  
* Part III (time permitting): Open qustions on Reinforcement Learning

 -->



<!-- The course will be graded with a final project -- consisting in an in-depth survey of a topic related to the syllabus,
plus a participation grade. The detailed abstract of the project will be graded at the mid-term.  -->


<!-- 
**Final Project is due May 1st by email to the instructors**

## Lectures

| Week        | Lecture Date           | Topic       |  References                     |
| ---------------|----------------| ------------|---------------------------|
| 1 | 1/28  | **Guest Lecture: Arthur Szlam (Facebook)**  |  [References](doc/refs.md#lec1)  |
| 2 | 2/4  | **Lec2** Euclidean Geometric Stability. [Slides](https://github.com/joanbruna/MathsDL-spring18/blob/master/lectures/lecture2.pdf) |  [References](doc/refs.md#lec2)  |
| 3 | 2/11  | **Guest Lecture: Leon Bottou (Facebook/NYU)** [Slides](https://github.com/joanbruna/MathsDL-spring18/blob/master/lectures/bottou-02.06.2018.pdf)  |  [References](doc/refs.md#lec3)  |
| 4 | 2/18  | **Lec3** Scattering Transforms and CNNs [Slides](https://github.com/joanbruna/MathsDL-spring18/blob/master/lectures/lecture3.pdf) |  [References](doc/refs.md#lec3)  |
| 5 | 2/25  | **Lec4** Non-Euclidean Geometric Stability. Gromov-Hausdorff distances. Graph Neural Nets [Slides](https://github.com/joanbruna/MathsDL-spring18/blob/master/lectures/lecture4.pdf)|  [References](doc/refs.md#lec4)  |
| 6 | 3/4  | **Lec5** Graph Neural Network Applications [Slides](https://github.com/joanbruna/MathsDL-spring18/blob/master/lectures/lecture5.pdf) |  [References](doc/refs.md#lec5)  |
| 7 | 3/11  | **Lec6** Unsupervised Learning under Geometric Priors. Implicit vs Explicit models. Optimal Transport models. Microcanonical Models. Open Problems [Slides](https://github.com/joanbruna/MathsDL-spring18/blob/master/lectures/lecture6.pdf)  |  [References](doc/refs.md#lec6)  |
| 8 | 3/18  | **Spring Break**  |  [References](doc/refs.md#lec8)  |
| 9 | 3/25  | **Lec7** Discrete vs Continuous Time Optimization. The Convex Case. [Slides](https://github.com/joanbruna/MathsDL-spring18/blob/master/lectures/lecture7.pdf)   |  [References](doc/refs.md#lec7)  |
| 10 | 4/1  | **Lec8** Discrete vs Continuous Time Optimization. Stochastic and Non-convex case [Slides](https://github.com/joanbruna/MathsDL-spring18/blob/master/lectures/lecture8.pdf) |  [References](doc/refs.md#lec8)  |
| 11 | 4/8  | **Lec9** Gradient Descent on Non-convex Optimization. [Slides](https://github.com/joanbruna/MathsDL-spring18/blob/master/lectures/lecture9.pdf) |  [References](doc/refs.md#lec9)  |
| 12 | 4/15  | **Lec10** Gradient Descent on Non-convex Optimization. Escaping Saddle Points efficiently. [Slides](https://github.com/joanbruna/MathsDL-spring18/blob/master/lectures/lecture10.pdf) |  [References](doc/refs.md#lec10)  |
| 13 | 4/22  | **Lec11** Landscape of Deep Learning Optimization. Spin Glasses, Kac-Rice, RKHS, Topology. [Slides](https://github.com/joanbruna/MathsDL-spring18/blob/master/lectures/lecture11.pdf) |  [References](doc/refs.md#lec11)  |
| 14 | 4/29  | **Lec12** **Guest Lecture: Behnam Neyshabur (IAS/NYU): Generalization in Deep Learning** [Slides](https://github.com/joanbruna/MathsDL-spring18/blob/master/lectures/lecture12_behnamneyshabur.pdf) |  [References](doc/refs.md#lec12)  |
| 15 | 5/6  | **Lec13** Stability. Open Problems. |  [References](doc/refs.md#lec13)  |



### Lab sessions / Parallel Curricula

### DistributionalRL: [Living document](https://docs.google.com/document/d/1bk6txed3bjvPBsWF26HD4xab-iUylHtA7vGXfP4QFew/edit?usp=sharing)

* Class 1: Basics of RL and Q learning
  * Required Reading:
    * [Sutton and Barto](http://incompleteideas.net/book/bookdraft2017nov5.pdf) (Ch 3, Ch 4, Ch 5, Ch 6.5)
      * The standard introduction to RL.  Focus in Chapter 3 on getting used to the notation we’ll use throughout the module, and an introduction to the Bellman operator and fixed point equations.  In Chapter 4 the most important idea is value iteration (and exercise 4.10 will ask you to show why iterating the Q function is basically the same algorithm).
      * Chapter 5 considers using full rollouts to estimate our value / Q function, rather than the DP updates.  Focus on the difference between on-policy and off-policy, which will be relevant to the final algorithm.
      * Including 6.5 is an introduction to Q-learning in practice, updating one state-action pair at a time (without worrying about function approximation yet).
    * [Contraction Mapping Theorem](https://www.math.ucdavis.edu/~hunter/book/ch3.pdf) (3.1)
      * We’ll need the notion of contractions repeatedly throughout the module.  Their essential property is a unique fixed point, and you should have a clear understanding of the constructive proof of this fixed point (don’t worry about the ODE applications).
  * Questions:
    * Exercise 3.14, Exercise 4.10 in S & B
    * Prove the Bellman operator contracts Q functions with regard to the infinity norm
    * What is a sanity-check lower bound on complexity for Q learning?  Why might this be infeasible for RL problems in the wild?


### NeuralODE: [Living document](https://docs.google.com/document/d/1GHvyCCZ3Ep-IWa5QSQ6NMtPsCJry9h0jOPues5iEIus/edit?usp=sharing)

* Class 6: Neural ODEs
  * Motivation: Let’s read the paper! 
  * Required Reading: 
    * [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366)
    * [A blog post on NeuralODEs](https://rkevingibson.github.io/blog/neural-networks-as-ordinary-differential-equations/)
  * Optional Reading:
    * A follow-up paper by the authors on scalable continuous normalizing flows: [Free-form Continuous Dynamics for Scalable Reversible Generative Models](https://arxiv.org/abs/1810.01367)
    
* Class 5: The adjoint method (and auto-diff)
  * Motivation: The adjoint method is a numerical method for efficiently computing the gradient of a function in numerical optimization problems. Understanding this method is essential to understand how to train ‘continuous depth’ nets. We also review the basics of Automatic Differentiation, which will help us understand the efficiency of the algorithm proposed in the NeuralODE paper.  
  * Required Reading: 
    * Section 8.7 from [Computational Science and Engineering](http://math.mit.edu/~gs/cse/) (CSE)
    * Sections 2,3 from [Automatic Differentiation in Machine Learning: a Survey](http://www.jmlr.org/papers/volume18/17-468/17-468.pdf)
  * Optional Reading:
    * [Prof. Steven G. Johnson's notes on adjoint method](http://math.mit.edu/~stevenj/notes.html)
  * Questions:
    * Exercises 1,2,3 from Section 8.7 of CSE
    * Consider the problem of optimizing a real-valued function g over the solution of the ODE y' = Ay , y(0) = y_0 at time T>0: min_{y0, A} g(y(T)). What is the solution of the adjoint equation?
    * How do you get eq. (14) in Section 8.7 of CSE?


* Class 4: Normalizing Flow
  * Motivation: In this class we take a little detour through the topic of Normalizing Flows. This is used for density estimation and generative modeling, and it is another model which can be seen a time-discretization of its continuous-time counterpart.
  * Required Reading: 
    * [Density Estimation by Dual Ascent of the Log-likelihood](https://math.nyu.edu/faculty/tabak/publications/CMSV8-1-10.pdf), minus Section 3 (DE)
    * [A family of non-parametric density estimation algorithms](https://math.nyu.edu/faculty/tabak/publications/Tabak-Turner.pdf) 
    * [A post on Normalizing flow](http://akosiorek.github.io/ml/2018/04/03/norm_flows.html)
  * Optional Reading:
    * [Variational Inference with Normalizing Flows](https://arxiv.org/pdf/1505.05770.pdf)
    * [High-Dimensional Probability Estimation with Deep Density Models](https://arxiv.org/pdf/1302.5125.pdf)
  * Questions:
    * In DE, what is the difference between t and t, i.e. what do they represent?
    * In DE, why does eq. (4.2) imply convergence t   as t ?
    * What is the computational complexity of evaluating a determinant of a N x N matrix, and why is that relevant in this context?


* Class 3: ResNets
  * Motivation: The introduction of Residual Networks (ResNets) made possible to train very deep networks. In this section we study some residual architectures variants and their properties. We then look into how ResNets approximates ODEs and how this interpretation can motivate neural net architectures and new training approaches. 
  * Required Reading: 
    * ResNets: [ResNets](https://www.coursera.org/lecture/convolutional-neural-networks/resnets-HAhz9) and [An Overview of ResNet and its Variants](https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035)
    * ResNets and ODEs: 
      * Sections 1-3 of [Multi-level Residual Networks from Dynamical Systems View](https://arxiv.org/pdf/1710.10348.pdf)
      * [Reversible Architectures for Arbitrarily Deep Residual Neural Networks](https://arxiv.org/abs/1709.03698)
  * Optional Reading:
    * The original ResNets paper: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
    * Another blog post on ResNets: [Understanding and Implementing Architectures of ResNet and ResNeXt for state-of-the-art Image Classification](https://medium.com/@14prakash/understanding-and-implementing-architectures-of-resnet-and-resnext-for-state-of-the-art-image-cf51669e1624)
    * Invertible ResNets: [The Reversible Residual Network: Backpropagation Without Storing Activations](https://arxiv.org/pdf/1707.04585.pdf)
    * [Stable Architectures for Deep Neural Networks](https://arxiv.org/pdf/1705.03341.pdf)
  * Questions:
    * Can you think of any other neural network architectures which can be seen as discretizations of some ODE?
    * Do you understand why adding ‘residual layers’ should not degrade the network performance?
    * How do the authors of (Multi-level […]) explain the phenomena of still having almost as good performances in residual networks when removing a layer?
    * Implement your favourite variant ResNet variant

* Class 2: Numerical solution of ODEs II
  * Motivation: In the previous class we introduced some simple schemes to numerically solve ODEs. In this class we go through some more involved schemes and their convergence analysis. 
  * Required Reading: 
    * Runge-Kutta methods: Section 11.8 from NM or Sections 12.{5,12} from NA
    * Multi-step methods: Sections 12.6-9 from NA or Section 11.5-6 from NM
    * System of ODEs: Sections 11.9-10 from NM or Sections 12.10-11 from NA
  * Optional Reading:
    * [Prof. Trefethen's class ODEs and Nonlinear Dynamics 4.1](http://podcasts.ox.ac.uk/odes-and-nonlinear-dynamics-41)
    * Predictor-corrector methods: Section 11.7 from NM
    * Richardson extrapolation: Section 16.4 from [Numerical Recipes](http://numerical.recipes/)
    * [Automatic Selection of Methods for Solving Stiff and Nonstiff Systems of Ordinary Differential Equations](https://epubs.siam.org/doi/pdf/10.1137/0904010?casa_token=sBjDZTSayFQAAAAA:XhlfyWkS4MRFNRnrZ6LmQff_UXAH7riLBkpcA58llDnYEJycmMMbMCli9cFkoYKRT7uNos94IpA)
  * Questions:
    * From NA, Section 12: Exercises 12.11, 12.12, 12.19


* Class 1: Numerical solution of ODEs I
  * Motivation: ODEs are used to mathematically model a number of natural processes and phenomena. The study of their numerical 
    simulations is one of the main topics in numerical analysis and of fundamental importance in applied sciences. 
  * Required Reading: 
    * Sections 12.1-4 from [An Introduction to Numerical Analysis](https://www.cambridge.org/core/books/an-introduction-to-numerical-analysis/FD8BCAD7FE68002E2179DFF68B8B7237#) (NA) or Sections 11.1-3 from [Numerical Mathematics](https://www.springer.com/us/book/9783540346586?token=holiday18&utm_campaign=3_fjp8312_us_dsa_springer_holiday18&gclid=Cj0KCQiAvebhBRD5ARIsAIQUmnlViB7VsUn-2tABSAhIvYaJgSEqmJXD7F4A7EgyDQtY9v_GeUsNif8aArGAEALw_wcB) (NM)
  * Optional Reading:
    * Section 12.5 from NM
    * [Prof. Trefethen's class ODEs and Nonlinear Dynamics 4.2](http://podcasts.ox.ac.uk/odes-and-nonlinear-dynamics-42)
  * Questions:
    * From NM, Section 11.12: Exercise 1 
    * From NA, Section 12: Exercises 12.3,12.4, 12.7
    * Consider the following method for solving y' = f(y):
           y_{n+1} = y_n + h*(theta*f(y_n) + (1-theta)*f(y_{n+1}))                             
      Assuming sufficient smoothness of y and f, for what value of 0 <= theta <= 1 is 
      the truncation error the smallest? What does this mean about the accuracy of 
      the method?
    * [Notebook](https://colab.research.google.com/drive/1bNg-RzZoelB3w8AUQ6mefRQuN3AdrIqX) -->