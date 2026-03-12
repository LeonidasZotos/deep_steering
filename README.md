# Deep Steering LLMs
Repository for the Deep Steering Project (Evaluation of Perspective Steering of LLMs)


The main pipeline so far looks like this:
1. `answer_questionnaire.py`: The LLM answers the questionnaire
2. `analyse_political_compass_results.py`: The LLM answers are analysised, producing political orientation plot and text file containing the result, both in the same folder where the LLM responses are. 

3. [to test]  `create_combined_overview.py` to produce a plot with the selected experiments, as defined in exp_configs.json. 


Tools: 
1. `interactive_lookup.ipynb`: Similar to `analyse_political_compass_results.py` and `create_combined_overview.py`, allows to quickly create a results overview of one or more sets of questionnaire results. 
2.  `librarian.ipynb`: Can be used to find the experiment ID of the results you are interested in, if they exist.