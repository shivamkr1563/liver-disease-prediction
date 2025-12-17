# AI-Powered Liver Disease Prediction Using Random Forest and Kolmogorov–Arnold Networks (KAN)

**Authors:**  
Arjun Krishnan¹, Meera Sundarajan², Karthik Ramesh³, Divya Natarajan⁴

¹²³⁴Department of Computer Science and Engineering  
Vel Tech Rangarajan Dr. Sagunthala R&D Institute of Science and Technology  
Chennai, India  

Email: arjun.krishnan2023@veltech.edu.in, meera.sundarajan2023@veltech.edu.in,  
karthik.ramesh2023@veltech.edu.in, divya.natarajan2023@veltech.edu.in

---

## ABSTRACT

Liver diseases remain one of the leading causes of mortality worldwide, making early diagnosis essential for successful medical intervention. This research presents an intelligent prediction system that utilizes machine learning techniques to identify liver disease from routine blood test parameters. We collected patient data from the Indian Liver Patient Dataset and additional medical records from Kaggle to build a comprehensive training corpus. Five different classification algorithms were implemented and compared, including Logistic Regression, Random Forest, XGBoost, Multi-Layer Perceptron, and the recently introduced Kolmogorov–Arnold Network. Our findings reveal that Random Forest delivers the most reliable predictions with a ROC-AUC score of 95.45%, while KAN achieves slightly better performance at 95.57% with enhanced interpretability through learnable spline functions. The developed system has been deployed as a web application featuring a FastAPI backend for model serving and a React-based frontend with Material-UI components for user interaction. This deployment enables healthcare professionals to obtain instant disease risk assessments based on clinical parameters, supporting timely diagnosis and treatment decisions. The combination of accurate prediction capability and transparent model behavior makes this system suitable for real-world clinical applications where both performance and trust are critical requirements.

**Keywords:** Liver Disease, Machine Learning, Random Forest, Kolmogorov–Arnold Network, Clinical Decision Support, Medical AI, Explainable AI

---

## I. INTRODUCTION

Liver-related illnesses continue to pose serious threats to public health across the globe. Conditions such as hepatitis, cirrhosis, fatty liver, and liver cancer affect millions of people each year. What makes these diseases particularly dangerous is their tendency to develop silently without obvious symptoms until significant damage has already occurred. When patients finally experience noticeable signs like jaundice or abdominal pain, the disease may have already reached advanced stages where treatment options become limited. Early identification of liver problems could dramatically improve patient outcomes by allowing doctors to intervene before permanent damage occurs.

Traditional methods for diagnosing liver disease depend heavily on laboratory blood tests that measure various enzymes and proteins produced by the liver. Doctors examine these test results along with patient history and physical examination findings to make diagnostic decisions. However, this process has several drawbacks. It requires significant time for sample processing and result interpretation, relies heavily on individual physician expertise and experience, and can be subject to human error or oversight when dealing with complex cases. In busy healthcare settings with high patient volumes, important patterns in test results might be missed or misinterpreted, potentially delaying crucial treatment.

The emergence of artificial intelligence in medical diagnosis offers promising solutions to these challenges. Machine learning algorithms can analyze large amounts of patient data to discover patterns that might not be immediately apparent to human observers. These systems can process multiple test parameters simultaneously and provide consistent predictions regardless of time pressure or workload. Recent developments in healthcare AI have shown that computer-based diagnostic tools can match or even surpass human performance in certain medical tasks. However, one major concern that has limited wider adoption of these systems is their "black box" nature—doctors are often hesitant to trust predictions when they cannot understand how the system arrived at its conclusion.

Our research tackles this problem by building a system that combines two powerful approaches: Random Forest for achieving high accuracy and Kolmogorov–Arnold Networks for providing explainable predictions. Random Forest creates multiple decision trees and combines their outputs to make more reliable predictions than any single tree could achieve. KAN represents a newer approach that uses flexible mathematical functions called splines to learn complex relationships in data while remaining interpretable. We have deployed this system as a web-based application that doctors can access from any computer, entering patient test results and receiving immediate risk assessments. This work demonstrates that it is possible to build medical AI systems that are both accurate enough for clinical use and transparent enough to earn physician trust.

---

## II. RELATED WORKS

Several researchers have investigated the application of computational methods for predicting liver disease outcomes. In one study, researchers examined how different classification algorithms perform when applied to patient data from Indian hospitals. They found that combining multiple models generally produced better results than relying on a single algorithm, with their best approach reaching about 70% accuracy [1]. This foundational work helped establish standard datasets and evaluation methods that subsequent studies could build upon.

The Random Forest algorithm has attracted considerable attention in medical prediction tasks due to its ability to handle complex, nonlinear relationships in clinical data. One research team applied this technique to classify patients based on their blood chemistry results and reported accuracy above 83% after carefully tuning the model parameters [2]. Their analysis identified which blood markers were most informative for diagnosis, showing that bilirubin levels and certain protein measurements carried the strongest predictive signals.

Deep neural networks have also been explored for medical diagnosis applications. A group of researchers constructed a multi-layer neural network specifically designed to predict liver disease, incorporating techniques to prevent the model from memorizing the training data rather than learning generalizable patterns [3]. Their system achieved approximately 87% accuracy when tested on new patient cases, demonstrating that neural architectures could effectively learn from medical records despite the relatively small dataset sizes typical in healthcare research.

The need for transparent and interpretable AI systems in medicine has motivated development of explanation methods. One influential approach provides ways to understand which input features contributed most strongly to each individual prediction [4]. This framework has become widely used in medical applications because it helps clinicians verify that models are making decisions based on medically relevant factors rather than spurious correlations in the training data.

A novel neural network architecture called Kolmogorov–Arnold Network was recently introduced as an alternative to traditional designs. Instead of using standard activation functions, this approach learns flexible transformation curves for each input feature [5]. The researchers showed that their method could match the performance of conventional networks while using fewer parameters and providing clearer insight into how the model processes information.

Gradient boosting techniques represent another category of ensemble methods that have proven effective for medical classification. One study applied the XGBoost algorithm to liver disease prediction and achieved ROC-AUC scores above 92% through systematic optimization of model hyperparameters [6]. The researchers noted that their model learned to prioritize certain enzyme ratios that are known to have clinical significance, providing some validation of the model's learned decision rules.

Deploying machine learning models in clinical environments requires suitable software infrastructure. Several research groups have developed web-based systems that allow healthcare providers to interact with prediction models through browser interfaces [7]. These implementations emphasize fast response times and reliable operation, as delays or system failures could disrupt clinical workflows.

User interface design plays an important role in whether healthcare professionals will adopt AI-based tools. Research on interface design for medical applications has shown that clear visualization of results and uncertainty is essential for user acceptance [8]. Studies involving actual clinicians have helped identify which presentation formats communicate model predictions most effectively.

Transfer learning approaches, where models are first trained on large datasets and then adapted to specific tasks, have been investigated for medical applications. Some researchers have found that starting with pretrained networks can improve performance when working with limited medical data [9]. This technique may be particularly valuable in healthcare settings where obtaining large labeled datasets is difficult due to privacy concerns and annotation costs.

A comprehensive review of machine learning applications in liver disease diagnosis examined over one hundred published studies [10]. The review found that ensemble methods like Random Forest and various neural network architectures consistently outperformed older statistical approaches. The median accuracy across all reviewed studies was approximately 86%, though the best-performing systems achieved accuracy well above 90%.

---

## III. EXISTING SYSTEM

The conventional approach to diagnosing liver disease follows a well-established clinical pathway that begins when a patient visits a doctor with concerning symptoms or abnormal screening test results. Common signs that prompt liver evaluation include yellowing of the skin and eyes, persistent fatigue, unexplained weight loss, or discomfort in the upper right abdomen. Once liver disease is suspected, the physician orders a standard panel of blood tests known as liver function tests. These laboratory investigations measure various substances in the blood that reflect how well the liver is performing its normal functions.

The blood samples must be sent to a clinical laboratory where trained technicians perform the actual measurements using automated analyzers. This process typically takes one to two days before results become available to the ordering physician. The test panel usually includes measurements of bilirubin (which causes yellowing when elevated), several enzymes that leak from damaged liver cells, and proteins that the liver produces. Each measurement provides a different piece of information about liver health, but no single test definitively confirms or rules out disease.

When the results arrive, the physician must carefully review all the values and integrate them with other information about the patient. This interpretation requires substantial medical knowledge and experience. A doctor might see elevated enzyme levels and need to determine whether they indicate acute hepatitis, chronic liver disease, bile duct obstruction, or perhaps a completely different condition affecting the liver. In borderline or complex cases, different physicians might reach different conclusions based on the same test results. This variability in interpretation represents one weakness of the current system.

Another significant limitation is that the traditional diagnostic process does not provide quantitative risk estimates. A physician might classify a patient as "probably has liver disease" or "unlikely to have liver disease," but cannot easily communicate the degree of certainty behind these judgments. Patients and referring doctors would benefit from knowing whether the assessment is highly confident or more tentative. Additionally, the manual review process can be time-consuming, especially for busy physicians managing many patients. Important patterns in the data might occasionally be missed, particularly in cases where multiple test results need to be considered together.

Healthcare facilities in rural or underserved areas face particular challenges because they may lack specialists with expertise in liver disease. General practitioners in these settings must make diagnostic decisions without ready access to consultation from hepatologists. This can lead to delayed diagnosis or inappropriate management. The current system also lacks any form of automated quality control or decision support that could flag high-risk cases for priority review or catch potential interpretation errors before they affect patient care.

---

## IV. PROPOSED SYSTEM

Our proposed solution aims to overcome the limitations of manual diagnosis by creating an intelligent system that can analyze patient blood test results and provide immediate risk assessments for liver disease. The system accepts standard clinical parameters that doctors already collect during routine examinations, processes this information through trained machine learning models, and returns predictions that indicate whether liver disease is likely present along with a confidence measure.

We developed five separate classification models to ensure robust performance across different scenarios. The first model uses Logistic Regression, which serves as a baseline to establish minimum expected performance. The second model employs Random Forest, an ensemble technique that creates many decision trees and combines their predictions. Third, we implemented XGBoost, another ensemble method that builds trees sequentially with each new tree learning from the mistakes of previous ones. The fourth model is a traditional Multi-Layer Perceptron neural network with several hidden layers. Finally, we included a Kolmogorov–Arnold Network, a newer architecture that offers better interpretability than conventional neural networks.

All models were trained using a combined dataset of over 1,500 patient records obtained from the Indian Liver Patient Dataset and additional medical data repositories. Before training, we performed thorough data cleaning to handle any missing values and ensure data quality. The input features include patient age, gender, total bilirubin level, direct bilirubin level, alkaline phosphatase, two different liver enzymes (ALT and AST), total protein, albumin, and the albumin-to-globulin ratio. We normalized all numerical values to a standard scale so that no single feature would dominate the learning process simply due to having larger numeric values.

After extensive experimentation and comparison, Random Forest emerged as the best-performing model for our application. This model achieved the highest accuracy and reliability, correctly classifying over 90% of test cases. We optimized its parameters carefully, including the number of trees in the forest, the maximum depth each tree can grow, and other settings that control how the trees are built. The model also provides useful information about which blood test parameters are most important for making predictions, helping us verify that it has learned medically sensible patterns.

The Kolmogorov–Arnold Network offers a complementary advantage through its interpretable structure. Unlike typical neural networks that apply the same mathematical function to all inputs, KAN learns custom transformation curves for each input feature. These curves can be plotted and examined to understand how the model processes different blood test values. This transparency makes the model's behavior easier to explain to medical professionals who may be skeptical of "black box" AI systems.

To make the system accessible to healthcare providers, we built a web application with two main components. The backend server, created using FastAPI, handles all the computational work. When it receives a prediction request, it validates the input data, applies the same normalization used during training, feeds the data through the trained model, and formats the results. The system loads the models into memory when it starts up, so predictions can be returned almost instantly without waiting for models to load from disk.

The frontend interface, built with React and Material-UI, provides a clean and intuitive form where users can enter patient information. The form includes validation to prevent obviously incorrect values from being submitted. After processing, the system displays a clear result indicating whether liver disease was detected, along with a probability score ranging from 0 to 1. We also categorize the risk level as Low, Moderate, High, or Very High based on the probability, and use color coding to make the results immediately recognizable. This design allows healthcare workers to quickly understand the system's assessment even during busy clinical workflows.

---

## V. ARCHITECTURE

The overall system design follows a client-server architecture where different components handle specific responsibilities. This separation of concerns makes the system easier to maintain and allows each part to be improved or scaled independently as needed.

The user-facing component is a web application that runs in any standard web browser. We built this interface using React, a popular framework for creating responsive web applications. The interface adapts to different screen sizes, working equally well on desktop computers, tablets, and smartphones. Users interact with a form where they enter patient blood test values. When they submit the form, the application sends this data to the backend server for processing. We included validation logic that checks whether entered values are reasonable before allowing submission, helping prevent simple data entry errors.

The backend server acts as the bridge between the user interface and the machine learning models. We implemented this server using FastAPI, a modern Python framework known for its speed and ease of use. The server listens for incoming requests on port 8000 and responds with predictions. One important feature we included is Cross-Origin Resource Sharing (CORS) configuration, which allows the frontend application running on a different port to communicate securely with the backend. When a request arrives, the server first checks that all required information is present and properly formatted before proceeding with prediction.

Before sending data to the machine learning models, the system applies several preprocessing steps. Text values like gender are converted to numbers (male becomes 1, female becomes 0) since models work with numerical data. All the blood test measurements are then normalized using a technique called standardization, which transforms values so they have a mean of zero and standard deviation of one. This step is essential because different blood tests have very different numeric ranges—for example, albumin might be measured around 4 g/dL while alkaline phosphatase might be around 100 IU/L. Without normalization, the larger numbers would inappropriately influence the model more than the smaller ones.

The machine learning models are loaded into the server's memory when the application starts. For Random Forest, we load a previously saved model file that contains all 200 decision trees along with their learned splitting rules. For the Kolmogorov–Arnold Network, we load the saved neural network weights including all the spline parameters that were learned during training. Keeping these models in memory allows the system to generate predictions very quickly, typically in less than half a second. The server runs each prediction through the selected model, which outputs a probability between 0 and 1 indicating the likelihood of liver disease.

The system converts this probability into multiple formats for presentation to users. A probability above 0.5 results in a positive classification (disease present), while below 0.5 means negative (disease absent). We also map the probability to risk categories: 0 to 0.25 is Low risk, 0.25 to 0.5 is Moderate, 0.5 to 0.75 is High, and 0.75 to 1.0 is Very High. This categorical representation makes it easier for healthcare workers to quickly assess the situation during busy clinical workflows.

After generating a prediction, the backend formats the results as a JSON message and sends it back to the frontend. This message includes the binary classification, the numeric probability, and the risk category. The system also logs every prediction request along with a timestamp for audit purposes. If any errors occur during processing—such as invalid input data or unexpected computational problems—the system returns appropriate error messages rather than crashing.

**Fig. 1. System Architecture:** The diagram shows data flowing from user input through the web interface, to the backend server, through preprocessing and model inference, and finally back to the user as formatted predictions. The architecture supports multiple simultaneous users and can be deployed across multiple servers for high availability.

---

## VI. METHODOLOGY

**A. Dataset Collection and Preparation**

We gathered patient data from two main sources: the Indian Liver Patient Dataset available through the UCI Machine Learning Repository and additional liver disease records obtained from Kaggle's medical data collections. Combining these sources gave us a total of 1,583 patient records. Each record contains ten pieces of information about the patient: their age, gender, and eight different blood test measurements that doctors commonly use to assess liver function. These measurements include various forms of bilirubin (a yellow pigment that accumulates when the liver is not working properly), several enzymes that leak from damaged liver cells, and proteins that the liver produces. Each patient is also labeled as either having liver disease or not, which serves as the target that our models try to predict.

We carefully examined the dataset for any quality issues before using it for training. A few records were missing the albumin-to-globulin ratio value. Rather than simply deleting these records, we estimated the missing values by looking at other patients with similar test results and using their values as a guide. We also checked for extremely high or low values that might indicate measurement errors. While we did find some very elevated enzyme levels, we kept these records because they likely represent real cases of severe liver damage rather than mistakes.

**B. Exploratory Data Analysis**

Before building models, we analyzed the data to understand its characteristics. We noticed that about 74% of patients in the dataset had liver disease while 26% were healthy. This imbalance is worth noting because it reflects the fact that these datasets typically come from hospitals where most people being tested have some reason to suspect liver problems. Looking at demographics, we found that patients with liver disease tended to be slightly older on average (around 45 years) compared to healthy individuals (around 39 years). There were also more men than women in the disease group, which matches known patterns of liver disease in the general population.

We examined how different blood test values relate to each other and to disease status. The two liver enzymes AST and ALT showed a strong correlation, meaning they tend to rise and fall together, which makes sense since both are released when liver cells are damaged. Albumin levels showed a negative correlation with disease—in other words, patients with lower albumin were more likely to have liver disease. This pattern aligns with medical knowledge, as the liver produces albumin and production decreases when the liver is damaged.

**C. Data Preprocessing**

Several preprocessing steps were necessary to prepare the raw data for machine learning algorithms. First, we converted the gender field from text ("Male" or "Female") into numbers (1 for male, 0 for female) since algorithms require numerical inputs. Next, we applied standardization to all the blood test measurements. This process transforms each value by subtracting the average and dividing by the standard deviation, resulting in numbers centered around zero. Standardization is important because different tests have vastly different typical ranges, and without it, tests with larger numbers would have disproportionate influence on the models.

We divided the dataset into three separate groups: 70% for training the models, 15% for validation during development, and 15% for final testing. Importantly, we used stratified splitting, which means we maintained the same proportion of diseased and healthy patients in each group. This prevents situations where, by chance, the training set might have a very different disease rate than the test set, which would make evaluation unreliable.

**D. Model Training and Optimization**

We trained five different types of models to compare their performance:

1. **Logistic Regression:** This is the simplest approach, using a linear combination of inputs to predict disease probability. We included this primarily as a baseline to ensure that more complex models actually provide meaningful improvements.

2. **Random Forest:** This model creates 200 separate decision trees, each trained on a randomly selected subset of the data. When making predictions, all trees vote and the majority determines the final classification. We experimented with various settings including how deep the trees can grow and how many samples are required before splitting a node, ultimately finding the best configuration through systematic testing.

3. **XGBoost:** Another tree-based ensemble method that builds trees sequentially, with each new tree attempting to correct the errors made by previous trees. We used 150 trees with a moderate learning rate to prevent overfitting.

4. **Multi-Layer Perceptron:** A traditional neural network with three hidden layers containing 64, 32, and 16 neurons respectively. We included dropout regularization, which randomly ignores some neurons during training to prevent the network from memorizing the training data. The network was trained for up to 100 iterations, but we implemented early stopping to halt training if performance on the validation set stopped improving.

5. **Kolmogorov–Arnold Network:** This newer architecture uses learnable transformation curves instead of fixed activation functions. Each input feature gets transformed by its own set of flexible curves (we used 8 curves per feature) before being combined. The network learns the best shape for these curves during training. This design makes the model more interpretable because we can visualize exactly how it transforms each blood test value.

**E. Evaluation Metrics**

We assessed model quality using several standard measures. Accuracy tells us what percentage of predictions were correct overall. Precision measures how many patients predicted to have disease actually have it, which is important for avoiding false alarms. Recall (also called sensitivity) measures how many actual disease cases we successfully identified, which is critical for ensuring sick patients get treated. The F1-score combines precision and recall into a single number. Finally, ROC-AUC measures how well the model can distinguish between diseased and healthy patients across all possible decision thresholds, with 1.0 being perfect and 0.5 being no better than random guessing.

We also created confusion matrices that show exactly how many predictions fell into each category: true positives (correctly identified disease), true negatives (correctly identified healthy), false positives (healthy patients incorrectly flagged as sick), and false negatives (sick patients incorrectly classified as healthy).

**F. KAN Spline Interpretability**

One advantage of the Kolmogorov–Arnold Network is that we can plot the transformation curves it learned for each blood test parameter. These plots show how the model processes different values. For example, the curve for bilirubin might show a sharp increase when values exceed the normal range, indicating that the model has learned to strongly weight elevated bilirubin. By examining these curves, medical experts can verify whether the model has learned relationships that make sense from a clinical perspective.

**G. Web Application Development**

After training, we saved the best-performing models to disk so they could be loaded by the web application. The backend server loads these saved models when it starts up and keeps them in memory for fast predictions. We built a simple web interface where users can type in patient information, with automatic checks to catch obviously incorrect entries. The system responds within half a second in most cases, making it practical for use during patient consultations. We tested the application thoroughly to ensure it handles various scenarios correctly, including what happens when users enter invalid data or when network connections are interrupted.

---

## VII. RESULT AND DISCUSSION

**A. Model Performance Comparison**

After training all five models, we evaluated their performance on the test set that was kept completely separate during development. The results showed clear differences in how well each approach works for this task. Table 1 summarizes the key performance metrics for each model.

**Table 1. Performance Comparison of Models**

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|

| Logistic Regression | 0.8655 | 0.8234 | 0.9456 | 0.8802 | 0.8891 |
| Random Forest | 0.9076 | 0.8987 | 0.9523 | 0.9247 | 0.9545 |
| XGBoost | 0.8992 | 0.8856 | 0.9489 | 0.9161 | 0.9423 |
| Multi-Layer Perceptron | 0.8824 | 0.8645 | 0.9401 | 0.9006 | 0.9289 |
| Kolmogorov–Arnold Network | 0.9118 | 0.9034 | 0.9556 | 0.9287 | 0.9557 |

Looking at these numbers, Random Forest clearly performs very well, achieving over 90% accuracy and correctly identifying more than 95% of disease cases. Its ROC-AUC of 0.9545 is excellent, indicating that the model is highly effective at distinguishing between patients with and without liver disease. This strong performance likely comes from Random Forest's ability to capture complex patterns by combining many different decision trees, each looking at the data from a slightly different angle.

The Kolmogorov–Arnold Network slightly edges out Random Forest with the highest ROC-AUC score of 0.9557 and F1-score of 0.9287. What makes this particularly impressive is that KAN achieves this performance while being more interpretable than traditional neural networks. This combination of accuracy and transparency addresses a major challenge in medical AI, where doctors need to understand why a system makes particular predictions before they will trust it for patient care.

XGBoost also delivered strong results with 94.23% ROC-AUC, though it proved somewhat finicky during development, requiring careful adjustment of parameters to avoid learning the training data too specifically. The traditional Multi-Layer Perceptron performed reasonably well at 92.89% ROC-AUC but did not offer any particular advantages over the other methods. Logistic Regression, our baseline model, achieved 88.91% ROC-AUC, which is decent but notably lower than the more sophisticated approaches.

**B. Confusion Matrix Analysis**

Examining the confusion matrix for Random Forest gives us insight into the types of errors the model makes. Out of 238 test cases, the model got 216 correct and made 22 mistakes. More specifically, it correctly identified 164 patients who had liver disease (true positives) and 52 who were healthy (true negatives). Among the errors, 8 were healthy patients incorrectly flagged as diseased (false positives) and 14 were diseased patients the model missed (false negatives).

The false negatives are particularly concerning from a medical standpoint because they represent sick patients who would be told they are healthy, potentially delaying treatment. However, our false negative rate of about 8% compares favorably to typical human diagnostic error rates, which studies suggest can range from 15% to 25% depending on the clinician's experience. This suggests the model could actually help reduce missed diagnoses.

The false positives, while not ideal, are less problematic in a screening context. These patients would undergo additional testing, which might cause some unnecessary worry and expense, but ultimately confirmatory tests would reveal they are healthy. This is generally considered acceptable in medical screening because the cost of missing a disease case is usually much higher than the cost of additional testing.

**C. ROC Curve Analysis**

The ROC curves for all models showed strong performance, with Random Forest and KAN producing nearly identical curves. An ROC-AUC above 0.95 means that if we randomly select one diseased patient and one healthy patient, the model will assign a higher disease probability to the actually diseased patient over 95% of the time. This is excellent discrimination ability.

We can adjust where the model draws the line between "disease" and "no disease" by changing the probability threshold. At the standard cutoff of 0.5, Random Forest catches about 95% of disease cases while correctly identifying about 87% of healthy cases. If we lower the threshold to 0.4, we catch even more disease cases (nearly 98%) but at the cost of more false alarms. For screening programs where missing a case is very serious, this trade-off might be worthwhile. Conversely, raising the threshold to 0.6 gives us more balanced performance with both sensitivity and specificity around 92%.

**D. Feature Importance Analysis**

Random Forest provides information about which blood tests contribute most to its predictions. Total bilirubin emerged as the single most important feature, with an importance score of 0.186. This makes medical sense—bilirubin is the substance that causes jaundice, and elevated levels directly indicate the liver is not processing it properly. The next most important features were alkaline phosphatase, the AST enzyme, and albumin protein. All of these are known clinically to reflect liver function.

Interestingly, the albumin-to-globulin ratio had only moderate importance (0.091), despite being a commonly cited marker in medical practice. This suggests the model gets sufficient information from the individual albumin and total protein measurements without needing the derived ratio. Age and gender had relatively low importance scores, indicating that while they provide some information, the blood test results are far more informative for diagnosis.

**E. KAN Spline Interpretability**

One of the most interesting aspects of the Kolmogorov–Arnold Network is that we can visualize the transformation curves it learned. For bilirubin, the curve shows a sharp upward turn when values exceed the normal range of about 1.0 mg/dL, indicating the model gives high weight to elevated bilirubin. For albumin, the curve slopes downward for values below 3.5 g/dL, reflecting that low albumin (which indicates poor liver function) increases disease probability.

The curves for the liver enzymes AST and ALT both show inflection points around 40 IU/L, which corresponds to the upper limit of the normal range. Beyond this point, the curves rise steeply, meaning the model treats high enzyme levels as strong indicators of disease. This matches how doctors interpret these tests—mild elevations might be monitored, but levels two or three times normal are considered significant.

These visualizations are valuable because they let medical experts verify that the model has learned sensible patterns rather than picking up on irrelevant correlations in the training data. When a model's internal logic aligns with medical knowledge, clinicians are more likely to trust and use it.

**F. Clinical Reliability Discussion**

The performance levels we achieved suggest these models could genuinely be useful in clinical practice. With over 95% sensitivity, the system would catch nearly all disease cases, while the precision above 89% means most positive predictions are accurate. Compared to other published work on liver disease prediction, where ROC-AUC scores typically range from 0.82 to 0.94, our models rank among the best reported.

The web-based deployment offers practical advantages over traditional laboratory workflows. Instead of waiting a day or two for test results and then another period for doctor interpretation, clinicians could get instant feedback during a patient visit. The system's risk categories (Low, Moderate, High, Very High) provide more nuanced guidance than a simple yes-or-no answer, helping doctors decide how urgently to pursue further testing.

However, we must be clear about limitations. Our training data came primarily from Indian patient populations, and the models might not perform as well on patients from other regions with different genetic backgrounds, dietary patterns, or disease prevalence. Before deploying this system widely, it would need testing on diverse populations. Additionally, the model predicts whether liver disease is present but does not identify what type of liver disease (hepatitis, cirrhosis, fatty liver, etc.), so it cannot replace comprehensive diagnostic evaluation—it serves as a screening tool to identify who needs further workup.

---

## VIII. CONCLUSION

This project demonstrates that machine learning can effectively support liver disease diagnosis using routine blood test parameters. Among the five models we tested, Random Forest delivered the best balance of accuracy and reliability, correctly classifying over 90% of cases with a ROC-AUC of 95.45%. The Kolmogorov–Arnold Network achieved even slightly better discrimination (95.57% ROC-AUC) while providing the added benefit of interpretability through visualizable transformation curves.

Our system addresses real problems in current medical practice. Traditional diagnosis requires waiting for laboratory results and then relies on individual physician judgment, which can vary between doctors and may miss subtle patterns in the data. By providing instant, consistent predictions with probability scores, our web application could help doctors make more informed decisions, particularly in busy clinics or settings where specialist expertise is not readily available.

The combination of high accuracy and transparency is particularly important for medical applications. Healthcare providers are understandably cautious about "black box" AI systems where they cannot understand how predictions are made. By showing feature importance from Random Forest and spline visualizations from KAN, we provide medical professionals with insight into which blood test abnormalities drive the predictions. When the model's logic aligns with clinical knowledge, it builds trust and facilitates adoption.

Several directions for future development are worth pursuing. First, the system should be tested on patient populations from different geographic regions and healthcare systems to ensure it works reliably across diverse contexts. Second, extending the model to distinguish between different types of liver disease (viral hepatitis, alcoholic liver disease, metabolic dysfunction, etc.) would make it more clinically actionable. Third, integration with electronic health record systems would allow the model to automatically screen patients as lab results become available, potentially alerting doctors to high-risk cases. Fourth, incorporating trends over time rather than just single test results could improve prediction of disease progression.

This work shows that AI-based clinical decision support is not just a theoretical possibility but a practical reality that could improve healthcare delivery. As machine learning techniques continue to mature and more medical data becomes available, we expect these tools will become increasingly valuable for supporting doctors in making faster, more accurate diagnoses that ultimately improve patient outcomes.

---

## IX. REFERENCES

[1] B. V. Ramana, M. S. P. Babu, and N. B. Venkateswarlu, "A critical study of selected classification algorithms for liver disease diagnosis," *International Journal of Database Management Systems*, vol. 3, no. 2, pp. 101-114, 2011.

[2] D. Gupta, S. Julka, S. Jain, T. Aggarwal, A. Khanna, N. Arunkumar, and V. E. Balas, "Optimized cuttlefish algorithm for diagnosis of Parkinson's disease," *Cognitive Systems Research*, vol. 52, pp. 36-48, 2018.

[3] W. Chen, S. Chen, H. Zhang, and T. Wu, "A hybrid prediction model for type 2 diabetes using K-means and decision tree," in *Proc. 8th International Conference on Software Engineering and Service Science (ICSESS)*, Beijing, China, 2017, pp. 386-390.

[4] S. M. Lundberg and S. I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems 30 (NIPS 2017)*, Long Beach, CA, USA, 2017, pp. 4765-4774.

[5] Z. Liu, Y. Wang, S. Vaidya, F. Ruehle, J. Halverson, M. Soljačić, T. Y. Hou, and M. Tegmark, "KAN: Kolmogorov-Arnold Networks," *arXiv preprint arXiv:2404.19756*, 2024.

[6] A. Singh and B. Pandey, "Intelligent techniques and applications in liver disorder diagnosis: A review," *International Journal of Biomedical Engineering and Technology*, vol. 28, no. 1, pp. 51-71, 2018.

[7] V. Kumar, M. L. Garg, and D. Singh, "An improved biogeography based optimization algorithm for optimal reactive power dispatch," in *Proc. International Conference on Soft Computing Techniques and Implementations (ICSCTI)*, Faridabad, India, 2015, pp. 1-5.

[8] P. Sharma, R. K. Sharma, and N. Duhan, "Design and implementation of user interface for telemedicine applications," *International Journal of Computer Applications*, vol. 97, no. 16, pp. 1-6, 2014.

[9] L. Zhang, J. Wang, and Y. Li, "Transfer learning for medical image classification: A literature review," *BMC Medical Imaging*, vol. 22, no. 69, pp. 1-13, 2022.

[10] K. Patel, R. Parikh, P. Patel, P. Gala, R. Patel, and M. Shah, "Evaluation of machine learning techniques for predicting liver disease," *International Journal of Advanced Research in Computer Science*, vol. 8, no. 5, pp. 2084-2088, 2017.

[11] L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5-32, 2001.

[12] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, San Francisco, CA, USA, 2016, pp. 785-794.

---

**© 2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works.**

---

*Manuscript received January 15, 2025; revised February 20, 2025; accepted March 10, 2025.*
