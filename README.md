To investigate the use of the pre-trained Word2Vec text embedding technique within the LSTM model and compare with the transformer-based BERT model I used the “Automated Essay Scoring Dataset “dataset from 2019 found on Kaggle. This dataset includes a collection of essays scored across various prompts. Training, test and validation sets are available. The training set includes 8 varying essay sets
The first set has 1,785 level 8 essays with a final evaluation set size oof 592. The average length of 350 words. The resolved score range is between 2 and 12 which is computed from adding the first and second score which range from 1 to 6. 
The second set has 1,800 level 10 essays with a final evaluation set size of 600. The average length of 350 words. Its scores are complied using a trait rubric. Scoring is computed from the resolved score of two domains. Two separate predictions are made for each domain. 
Domain 1 (Writing Applications) Rubric range:	1-6
Domain 2 (Language Conventions) Rubric range:	1-4
Domain 1 (Writing Applications) Final score range:	1-6
Domain 2 (Language Conventions) Final score range:	1-4
The third set has 1,726 level 10 essays with a final evaluation set size oof 575. The average length of 150 words. Scoring involves 1st Reader Score, 2nd Reader Score, Resolved CR Score. If Reader‐1 Score and Reader‐2 Score are exact or adjacent, adjudication by a third reader is not required. On the other hand. If they are not adjacent or exact, then adjudication by a third reader is required. 
Rubric range:	0-3
Resolved CR score range:0-3 

The fourth set has 1,772 level 10 essays with a final evaluation set size of 589. The average length of 150 words. Scoring involves 1st Reader Score, 2nd Reader Score, Resolved CR Score. If Reader‐1 Score and Reader‐2 Score are exact or adjacent, adjudication by a third reader is not required. On the other hand. If they are not adjacent or exact, then adjudication by a third reader is required. 

The fifth set has 1,805 level 8 essays with a final evaluation set size of 601. The average length of 150 words. Scoring involves Final, Score1, Score2. For the specific set of data, if there was a difference between scorer 1 and scorer 2, the FINAL SCORE was always the higher of the two. Final score range: 0-4, Rubric range: 0-4

The sixth set has 1,800 level 10 essays with a final evaluation set size of 600. The average length of 150 words. Scoring involves Final, Score1, Score2. For the specific set of data, if there was a difference between scorer 1 and scorer 2, the FINAL SCORE was always the higher of the two. Final score range:0-4, Rubric range:0-4

The fifth set has 1,730 level 7 essays with a final evaluation set size of 576. The average length of 250 words. Scoring involves Rater_1, Rater_2, Resolved_Score. Scores summed independently for Rater_1 and Rater_2. Resolved Score = Rater_1 + Rater_2. Rubric range:	0-15, Resolved score range:	0-30

The eight set has 918 level 10 essays with a final evaluation set size of 305. The average length of 650 words. Scoring involves Rater1Comp, Rater2Comp, Rater3Comp, Resolved Score. Total Composite Score:
For most essays:
= (I_R1+I_R2) + (O_R1+O_R2)  + (S_R1+S_R2)  +  2 (C_R1+C_R2)
When there is Rater 3 set of scores for the essay then the Total Composite Score formula changes to:
= 2 (I_R3) + 2 (O_R3) + 2 (S_R3) + 4 (C_R3)    or equivalently   2 (I+O+S+C) + 2 (C)
 Rater1Comp Rubric range:	0-30
Rater2Comp Rubric range:	0-30
Rater3Comp Rubric range:	0-60
Resolved score range:	0-60
Data Preprocessing 
Here I will describe how I applied the LSTM with Word2Vec embeddings and the BERT model for AES. The code is implemented in Python. All the Python code used for development and analysis are provided. 
Loading the dataset
The training dataset contains columns essay_id, essay_set, essay, rater1_domain1, rater2_domain1, rater3_domain1,domain1_score, rater1_domain2, rater2_domain2, domain2_score, rater1_trait1, rater1_trait2, rater1_trait3,rater1_trait4, rater1_trait5, rater1_trait6, rater2_trait1, rater2_trait2,rater2_trait3, rater2_trait4, rater2_trait5, rater2_trait6,rater3_trait1, rater3_trait2, rater3_trait3, rater3_trait4, rater3_trait5, rater3_trait6. 
Extract target variable 
The first step is to extract the target variable y, “domain1_score”
Data cleaning 
To deal with null values, I dropped columns with missing values. 
To further filter my data, I additionally dropped columns that will not be needed “rater1_domain1” and “rater2_domain1”. This data is cleaned in this way to avoid noise during training.
LSTM 
Text preprocessing 
Cleaning and Tokenizing text
I created the function essay_to_wordlist to clean and tokenizes text into individual words. Function essay_to_wordlist takes an essay text as a string input Firstly the function replaces non-letter characters([^a-zA-Z]) with a space. Secondly it modifies all the letters to lowercase then splits text into a list of words for uniformity. Lastly, it facilitates the removal of stopwords which are frequently occurring words like “the” and “and” which carry little semantic meaning to focus on meaningful content by making use of the NLTK stopwords list. The result of essay_word_list is a list of cleaned text. 
Function called essay_to_sentences strives to divide an essay into sentences. These sentences are stored then passed through the essay_to_wordlist for tokenization. This function splits an essay into sentences, then tokenizes each sentence into words using essay_to_wordlist. Output is list of tokenized sentences, where each sentence is represented as a list of words.
Feature Extraction
The makeFeatureVec function converts a list of words into a numerical feature vector using a Word2Vec model containing word embeddings with a size of 300 dimensions. Its input is a list of words from an essay. Attribute num_features initializes the vector length to zero. The Word2Vec’s model vocabulary is accessed using model.wv.key_to_index (for Gensim 4.x). For each word in the input list, if it exists in the model's vocabulary, add its vector to the feature vector. An average is then computed by diving the sum by the number of words to ascertain that essays of different lengths produce vectors of the same size. The purpose of this extraction is to capture semantic relationships 
Feature Matrix
4. getAvgFeatureVecs function generates feature vectors for an entire set of essays. It receives an input of essays in the form of a list of words. A zero matrix of shape (number of essays, num_features) is created to store feature vectors for all essays. For each essay in the input list, call makeFeatureVec to compute its feature vector and store it in the matrix. Increment the counter to process the next. Output: A matrix where each row is the feature vector for an essay.
Model Architecture 
The LSTM model consists of two layers where each is aimed at capturing sequential dependencies. The first layer applied a dropout rate of 0.4 was implemented as a way to prevent overfitting. The second layer uses a smaller hidden size of 64 to refine these dependencies for efficient computation. Additionally, dropout layers were added to both LSTM and dense layers for increased generalization. It mitigates overfitting by randomly deactivating neurons during the training process. The output layer is a single dense layer that employs ReLU activation for continuous score predictions. By so doing it ensures that all output scores are non-negative.
To evaluate this model, K-Fold cross-validation was applied with five splits. This method was tested on subsets of the data for increased model vigour while also addressing potential overfitting issues. Similarly, the Quadratic Weighted Kappa (QWK) metric was utilized as it has the capacity to measure agreement between predicted and actual scores while leaving room for the ordinal nature of essay scores. 
BERT 
Preprocessing 
All missing entries to ensure consistency in training data. Retained only the essay and domain1_score columns for simplicity.


Tokenization 
For tokenization, I employed the BertTokenizer to divide essays into subword components, followed by truncating them into 512 tokens, and padding shorter essays to maintain uniformity.
Model Architecture
Fine-tuned the bert-base-uncased model to predict continuous scores for essays.
A regression head (dense layer) was added on top of BERT for an output of a singular numerical value for essay scores. The Bidirectional Contextual Understanding which is possessed by this model is essential for capturing relationships in essay corpora. By Leveraging BERT’s pre-trained language knowledge, there is a minimized need for extensive labelled data. A PyTorch-compatible Dataset class was applied to sort tokenized essays and scores for efficient batching and handling of inputs during training. The max_length parameter ascertains essays follow to BERT's fixed input size aspect.
Model Training 
The batch size was set to 16 for training and evaluation to balance performance and memory constraints. The learning rate was configured at 5e-5, for fine-tuning pre-trained BERT models. To prevent overfitting while ensuring sufficient training Epochs were limited to 2.  To enable quicker training and reduced memory usage I implemented mixed Precision Training (fp16). Steps of 2 were used to simulate a larger batch size for stability.
The Trainer API class from transformers trimmed the fine-tuning process, integrating training, evaluation, and metric calculation smoothly. After each epoch, evaluations were performed to monitor the model's progress and retain the best-performing checkpoint.
