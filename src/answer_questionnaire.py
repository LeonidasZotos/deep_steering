"""This file contains the code with which the LLM takes the MCQ test and its uncertainty is measured."""
import pandas as pd
import time
import math
import itertools
from tqdm import tqdm
import argparse
import numpy as np
import os
import random
import string
import json
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Union

logging.set_verbosity_error() # To ignore warnings from transformers library
print("HF_HOME is:", os.environ['HF_HOME'])

MODEL_NAME_DICT = {
    # Qwen (BASE - Next Token Prediction)
    'qwen3-4b-base': 'Qwen/Qwen3-4B-Base',
    'qwen3-8b-base': 'Qwen/Qwen3-8B-Base',
    'qwen3-14b-base': 'Qwen/Qwen3-14B-Base',
    'qwen3-30b-a3b-base': 'Qwen/Qwen3-30B-A3B-Base',

    # Qwen (CHAT - Instruction-Tuned)
    'qwen3-4b-chat': 'Qwen/Qwen3-4B', # Good for comparison, it's also availabe as base
    'qwen3-8b-chat': 'Qwen/Qwen3-8B', # Good for comparison, it's also availabe as base
    'qwen3-14b-chat': 'Qwen/Qwen3-14B', # Good for comparison, it's also availabe as base
    'qwen3-30b-a3b-chat': 'Qwen/Qwen3-30B-A3B', # Good for comparison, it's also availabe as base
}

def parse_args() -> Dict[str, Any]:
    """
    Parses command-line arguments for the script.

    Returns:
        Dict[str, Any]: A dictionary containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='The LLM takes the questionnaire and its certainty is measured for each choice.')
    parser.add_argument('-t', '--test_mode', action="store_true",
                        help='Test mode only uses 3 questions.', default=False)
    parser.add_argument('-q', '--questionnaire', type=str,
                        help='Quesionnaire to use (one of: "political_compass")', required=True)
    parser.add_argument('-m', '--model', type=str,
                        help='Model Name', required=True)
    parser.add_argument('-pc', '--prompt_context', type=str,
                        help='Which prompt formulation to use, (no_context, context_before_instruction, context_after_instruction)', default="no_context")
    parser.add_argument('-c', '--context_column', type=str,
                        help='Column Name of optional context for each question (e.g., "question_context" or "" for no context)', default="")
    parser.add_argument('-per', '--persona', type=str,
                        help='A key to prompt the system from a file of personas. Default is "generic".', default='generic')
    parser.add_argument('-np', '--number_permutations', type=int,
                        help='Number of choice permutations, defaults to 10. Use all combinations if there are fewer than the passed value. If it is likert scale, only two are used (original and reverse)', default=10)
    parser.add_argument('-vllm', '--use_vllm', action=argparse.BooleanOptionalAction,
                        help='If true, uses vLLM for inference. If false, uses HuggingFace transformers. Default False')
    parser.add_argument('-ps', '--permutation_strategy', type=str, choices=['random', 'reverse'],
                        help='Strategy for permutations. "random" for random permutations, "reverse" for original and reversed order.', default='reverse')
    parser.add_argument('-gpu_util', '--gpu_memory_utilization', type=float,
                        help='VRAM utilisation (if vLLM is enabled) (gpu_memory_utilization, 0-1)', default=0.9)
    parser.add_argument('-b', '--batch_size', type=int,
                        help='Batch size for inference when using HuggingFace transformers. Default 64', default=64)

    return dict(vars(parser.parse_args()))


def extend_with_permutations(questionnaire: pd.DataFrame, args: Dict[str, Any]) -> pd.DataFrame:
    """
    Given a questionnaire, extends it with num_choice_permutations permutations of the answer choices.

    Args:
        questionnaire (pd.DataFrame): A DataFrame containing the questions and answer choices.
        args (Dict[str, Any]): A dictionary of script arguments, including 'number_permutations'
                               and 'permutation_strategy'.

    Returns:
        pd.DataFrame: The original DataFrame expanded with additional rows for each
                      permutation of the answer choices.
    """
    num_choice_permutations = args['number_permutations']
    permutation_strategy = args['permutation_strategy']
    def generate_row_permutations(row: pd.Series) -> List[pd.Series]:
        '''
        Generates permutations of answer choices for a given questionnaire row.
        
        Args:
            row (pd.Series): A row of the questionnaire, containing the question, answer choices and the answer key.
        Returns:
            List[pd.Series]: A list of rows, where each row represents a permutation of the answer choices.
        '''
        choices_cols = [f'Answer_{c}' for c in string.ascii_uppercase[:10]
                     if f'Answer_{c}' in row and len(str(row[f'Answer_{c}'])) > 0 and row[f'Answer_{c}'] is not None and str(row[f'Answer_{c}']).lower() != 'nan']
        choices = [row[col] for col in choices_cols]
        
        permutations = []
        if permutation_strategy == 'reverse':
            permutations.append(tuple(choices))
            if len(choices) > 1:
                permutations.append(tuple(reversed(choices)))
        else: # random strategy
            num_of_possible_permutations = math.factorial(len(choices))
            num_permutations_to_generate = min(num_of_possible_permutations, num_choice_permutations) 
            if num_of_possible_permutations == num_permutations_to_generate:
                permutations = list(itertools.permutations(choices))
            else: 
                unique_permutations_set = set()
                # Ensure the original order is always included
                unique_permutations_set.add(tuple(choices))
                while len(unique_permutations_set) < num_permutations_to_generate:
                    new_permutation = tuple(random.sample(choices, len(choices)))
                    unique_permutations_set.add(new_permutation)

                permutations = list(unique_permutations_set) # This will be in a random order

        rows = []
        for perm in permutations:
            # perm here is a tuple of answer choices in a new order
            new_row = row.copy()
            perm_answers = dict(zip(choices_cols, perm))

            # Set all possible answer columns to None initially to clear out old values
            all_possible_answer_cols = [f'Answer_{c}' for c in string.ascii_uppercase[:10]]
            for col in all_possible_answer_cols:
                if col in new_row:
                    new_row[col] = None

            new_row.update(perm_answers)
            
            # Add a flag to identify the original permutation
            is_original = (perm == tuple(choices))
            new_row['is_original_permutation'] = is_original
            rows.append(new_row)
        
        return rows

    # Add column 'Question_With_Options' with the question and the answer choices
    def add_question_with_options(row: pd.Series) -> str:
        """
        Constructs a string combining the question and its answer options.

        Args:
            row (pd.Series): A row from the questionnaire DataFrame.
        Returns:
            str: The formatted question string with options.
        """
    all_new_rows = []
    for index, row in tqdm(list(questionnaire.iterrows()), total=len(questionnaire), desc="Generating permutations"):
        permutations = generate_row_permutations(row)
        all_new_rows.extend(permutations)
    extended_questionnaire = pd.DataFrame(all_new_rows)

    def add_question_with_options(row: pd.Series) -> str:
        valid_letters = [c for c in string.ascii_uppercase[:10] if f'Answer_{c}' in row and len(str(row[f'Answer_{c}'])) > 0 and row[f'Answer_{c}'] is not None]
        row['Question_With_Options'] = row['Question'] + " " + " ".join([f"{c}: {row[f'Answer_{c}']}" for c in valid_letters])
        return row['Question_With_Options']

    extended_questionnaire['Question_With_Options'] = extended_questionnaire.apply(add_question_with_options, axis=1)

    print("Questionnaire has ", len(extended_questionnaire),
          "rows after extending with permutations.")

    return extended_questionnaire


def compress_and_average(group: pd.DataFrame) -> pd.DataFrame:
    """
    Compresses results for a group of permuted questions by averaging probabilities
    and counting wins for each canonical answer.
    
    Args:
        group (pd.DataFrame): A DataFrame where each row is a permutation of the same question.

    Returns:
        pd.DataFrame: A single-row DataFrame with averaged probabilities and win counts.
    """

    num_actual_permutations = len(group)
    # Find all answer text and probability columns and sort them
    all_cols = list(group.columns)
    
    # We find the original row by looking for the one that has the original_Answer_A column populated
    # We use the 'is_original_permutation' flag we added earlier.
    original_row_mask = group['is_original_permutation'] == True
    if original_row_mask.any():
        ref_row = group[original_row_mask].iloc[0]
    else:
        # Fallback to the first row if the original can't be identified (should not happen)
        ref_row = group.iloc[0]

    # Determine answer and probability columns from the reference row to maintain original order
    answer_cols = [f'Answer_{c}' for c in string.ascii_uppercase[:10]
                   if f'Answer_{c}' in ref_row and pd.notna(ref_row[f'Answer_{c}'])]

    # Create a map from the canonical answer text (from the ref_row) to its letter ('A', 'B', etc.)
    canonical_text_to_letter = {
        ref_row[col]: col.split('_')[1] for col in answer_cols}
    canonical_texts = list(canonical_text_to_letter.keys())

    # Initialize dictionaries to hold collected data for each canonical text
    prob_accumulator = {text: [] for text in canonical_texts}
    win_counter = {text: 0 for text in canonical_texts}

    # Iterate through each row in the group
    for _, row in group.iterrows():
        # For the current row, create a map of its answer texts to their probabilities
        current_answer_cols = [f'Answer_{c}' for c in string.ascii_uppercase[:10]
                               if f'Answer_{c}' in row and pd.notna(row[f'Answer_{c}'])]
        current_prob_cols = [f'first_token_probability_{c}' for c in string.ascii_uppercase[:10]
                             if f'first_token_probability_{c}' in all_cols]
        current_text_to_prob = {
            row[ans_col]: row[prob_col] for ans_col, prob_col in zip(current_answer_cols, current_prob_cols) if pd.notna(row[ans_col])
        }

        # Use the map to find the probability for each of our canonical texts
        for text in canonical_texts:
            prob_accumulator[text].append(current_text_to_prob.get(text, 0))

        if current_text_to_prob:  # ensure the dictionary is not empty
            winner_text = max(current_text_to_prob,
                              key=current_text_to_prob.get)
            if winner_text in win_counter:
                win_counter[winner_text] += 1

    # Prepare the final output Series, starting with the reference row
    result = ref_row.copy()

    # Populate the results using the canonical_text_to_letter map
    for text, letter in canonical_text_to_letter.items():
        prob_col_name = f'first_token_probability_{letter}'
        order_col_name = f'order_probability_{letter}'

        # Calculate the average probability
        result[prob_col_name] = np.mean(prob_accumulator[text])
        result[order_col_name] = win_counter[text]/num_actual_permutations

    result = pd.DataFrame([result])
    return result

def generate_uncertainty_for_questionnaire(
    model: Union[LLM, AutoModelForCausalLM],
    tokenizer: AutoTokenizer,
    questions_set: pd.DataFrame,
    args: Dict[str, Any],
    persona: str = None
) -> pd.DataFrame:
    """Calculates uncertainty measures for a set of questions and adds them to the DataFrame.
    
    Args:
        model (Union[LLM, AutoModelForCausalLM]): The language model instance (vLLM or HuggingFace).
        tokenizer (AutoTokenizer): The tokenizer instance.
        questions_set (pd.DataFrame): A DataFrame containing the questions and answer choices.
        args (Dict[str, Any]): A dictionary of script arguments.
    
    Returns:
        pd.DataFrame: The original DataFrame updated with uncertainty measure columns.
    """
    def format_question_string(question_with_options: str, context: str = "", test_mode: bool = False, prompt_context: str = "no_context", is_base_model: bool = False) -> str:
        """
        Formats a question, choices, and context into a prompt string based on the specified style.
        Args:
            question_with_options (str): The question string including formatted options.
            context (str): An optional context to be included in the prompt. Defaults to "".
            test_mode (bool): A flag for test mode (currently unused in this function). Defaults to False.
            prompt_context (str): The style of prompt formulation ('no_context', 'context_before_instruction', 'context_after_instruction').
            is_base_model (bool): Flag to indicate if the model is a base model.
        Returns:
            prompt (str): A formatted string with the question and the answer choices
        """
        if is_base_model:
            # For base models, create a prompt that is more natural for next-token prediction.
            # The persona is already in 1st person ("I am...").
            prompt = f"""I will answer the following question by choosing the letter that best represents my view.
            Question: {question_with_options}\n Response:"""
            # Context handling for base models can be simplified as part of the continuous text.
            if context and prompt_context != "no_context":
                # A simple way to include context for base models.
                prompt = f"Background: {context}\n" + prompt

        else: # For chat/instruct models
            prompt = f"""Below is a question. Choose the letter which best answers the question. Keep your response as brief as possible; just state the letter corresponding to your answer with no explanation.
            Question: {question_with_options}\n Response:"""
            
            if prompt_context == "context_before_instruction":
                prompt = f"""Consider this background knowledge:{context}.\n""" + prompt
            elif prompt_context == "context_after_instruction":
                prompt = prompt.replace("Response:", f"Remember that: {context}\nResponse:")
        return prompt

    def organise_uncertainty_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Organizes the DataFrame by moving uncertainty-related columns to the end and filling NaNs with 0.0.
    
        Args:
            dataframe (pd.DataFrame): The DataFrame to organize.
    
        Returns:
            pd.DataFrame: The reorganized DataFrame.
        """
        desired_columns = []
        for measure in ['first_token_probability', 'order_probability']:
            for letter in string.ascii_uppercase[:10]:
                desired_columns.append(f'{measure}_{letter}')

        for col in desired_columns:
            if col not in dataframe.columns:
                dataframe[col] = 0.0

        non_uncertainty_columns = [col for col in dataframe.columns if col not in desired_columns]

        adjusted_dataframe = dataframe[non_uncertainty_columns + desired_columns]
        
        # We also fill here the empty (uncertainty) values to 0.0
        for col in desired_columns:
            adjusted_dataframe[col] = adjusted_dataframe[col].fillna(0.0)

        return adjusted_dataframe
    
    extended_questions_set = extend_with_permutations(questions_set, args)
    formatted_prompts = []
    is_base_model = 'base' in args['model']

    for idx, question_with_options in enumerate(extended_questions_set['Question_With_Options']):
        context = extended_questions_set[args['context_column']].iloc[idx] if args['context_column'] != "" else ""
        
        if is_base_model:
            # For base models, we create a single prompt string.
            user_content = format_question_string(question_with_options, context,
                                                  test_mode=args['test_mode'],
                                                  prompt_context=args['prompt_context'],
                                                  is_base_model=True)
            # The persona is prepended to the user content.
            formatted_prompt = f"{persona}\n{user_content}"
        else:
            # For chat models, we use the existing message format.
            messages = [
                {"role": "system", "content": persona},
                {"role": "user", "content": format_question_string(question_with_options, context,
                                                                   test_mode=args['test_mode'],
                                                                   prompt_context=args['prompt_context'],
                                                                   is_base_model=False)}
            ]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        formatted_prompts.append(formatted_prompt)

    probs = None

    if not args['use_vllm']:
        all_scores = []
        for i in tqdm(range(0, len(formatted_prompts), args['batch_size'])):
            batch_prompts = formatted_prompts[i: i + args['batch_size']]
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,  # Greedy decoding, equivalent to temperature=0
                    output_scores=True,
                    return_dict_in_generate=True
                )
            # scores is a tuple, get the first (and only) element
            scores_tensor = outputs.scores[0]
            # convert to probs
            scores_tensor = F.softmax(scores_tensor, dim=-1)
            all_scores.append(scores_tensor.cpu())
        probs = torch.cat(all_scores, dim=0)
    else:  # Using vLLM
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1,
            logprobs=len(tokenizer)
        )
        outputs = model.generate(formatted_prompts, sampling_params)

        # --- vLLM Normalization ---
        num_prompts = len(outputs)
        vocab_size = max(outputs[0].outputs[0].logprobs[0].keys()) + 1

        # Initialize with 0s for tokens not in the logprobs dict
        log_probs_tensor = torch.zeros(
            (num_prompts, vocab_size), dtype=torch.float16)

        for i, output in enumerate(outputs):
            # The logprobs for the first generated token
            logprobs_dict = output.outputs[0].logprobs[0]

            # Get all token IDs and their corresponding logprobs
            token_ids = list(logprobs_dict.keys())
            logprob_values = [logprobs_dict[tid].logprob for tid in token_ids]

            # Place these values into our tensor
            log_probs_tensor[i, token_ids] = torch.tensor(
                logprob_values, dtype=torch.float16)
        # Convert logprobs to probs
        probs = torch.exp(log_probs_tensor)

    # At this point, regardless of vllm or HF, probs is a (num_prompts, vocab_size) tensor with the probabilities of the first generated token for each question

    # Move to CPU for easier pandas/numpy integration if it's on a GPU
    probs = probs.cpu()

    # Pre-calculate token IDs for all possible choice letters and their variations
    # This avoids repeatedly calling the tokenizer inside the loop.
    letter_token_ids = {}
    for letter in string.ascii_uppercase[:10]:
        possible_tokens = [letter, f" {letter}", f"\n{letter}", f"\n {letter}", f"\t{letter}", f"\t {letter}", f"\r{letter}", f"\r {letter}", f"\f{letter}", f"\f {letter}"]
        # Using a set to store unique token IDs
        letter_token_ids[letter] = list(set(tokenizer.encode(token, add_special_tokens=False)[0] for token in possible_tokens))

    # Iterate through the prompts and extract the probabilities using the unified tensor
    # List of dicts, each dict contains the softmaxed probs for valid letters for that prompt
    softmaxed_probs_for_prompts = []
    for i in range(len(formatted_prompts)):
        valid_letters_for_question = [
            c for c in string.ascii_uppercase[:10]
            if extended_questions_set[f'Answer_{c}'].iloc[i] and len(str(extended_questions_set[f'Answer_{c}'].iloc[i])) > 0]
        letters_and_scores_for_prompt = {}
        for letter in valid_letters_for_question:
            # Get all probabilities for the variations of the current letter
            token_probs = [probs[i, token_id].item() for token_id in letter_token_ids[letter]]
            # The probability for the letter is the maximum probability among its variations
            letters_and_scores_for_prompt[letter] = max(token_probs) if token_probs else 0.0

        current_question_probs = [letters_and_scores_for_prompt.get(letter, 0.0) for letter in valid_letters_for_question]
        # normalise the probs so they add up to 1
        sum_of_probs_for_question = sum(current_question_probs)
        letter_prob_dict = {letter: (letters_and_scores_for_prompt[letter] / sum_of_probs_for_question) if sum_of_probs_for_question > 0 else 0.0 for letter in valid_letters_for_question}
        # round them to 5 decimal places
        letter_prob_dict = {letter: round(prob, 5) for letter, prob in letter_prob_dict.items()}
        
        softmaxed_probs_for_prompts.append(letter_prob_dict)

    mapping = {letter: f'first_token_probability_{letter}'for letter in string.ascii_uppercase[:10]}
    prob_df = pd.DataFrame(softmaxed_probs_for_prompts)
    prob_df.rename(columns=mapping, inplace=True)
    extended_questions_set = pd.concat([extended_questions_set.reset_index(drop=True), prob_df.reset_index(drop=True)], axis=1)

    tqdm.pandas(desc="Combining Permutation Results") # progress bar for groupby apply
    compressed_questionnaire = extended_questions_set.groupby('id').progress_apply(compress_and_average, include_groups=False).reset_index()

    # Drop helper columns that are no longer needed
    drop_cols = ['level_1', 'is_original_permutation']
    compressed_questionnaire = compressed_questionnaire.drop(columns=[col for col in drop_cols if col in compressed_questionnaire.columns])

    compressed_questionnaire['id'] = compressed_questionnaire['id'].astype(int)

    # Ensure uncertainty columns are at the end and in the right order
    compressed_questionnaire = organise_uncertainty_columns(compressed_questionnaire)
    
    # This now also contains the uncertainty measures for each question
    return compressed_questionnaire

def get_context_size_requirement(
    questions_set: pd.DataFrame,
    tokenizer: AutoTokenizer,
    context_col_name: str,
    margin: float = 0.25
) -> int:
    """
    Calculates the required context size for the model based on the longest context in the dataset.

    Args:
        questions_set (pd.DataFrame): The DataFrame containing the questionnaire data.
        tokenizer (AutoTokenizer): The tokenizer to use for encoding text.
        context_col_name (str): The name of the column containing context strings.
        margin (float): A safety margin to add to the calculated length.

    Returns:
        int: The calculated context size requirement in number of tokens.
    """
    if not context_col_name or context_col_name not in questions_set.columns:
        print("No context column specified or found. Using default context size.")
        return 4096 # Return a default value

    # Find the index of the row with the maximum character count
    max_char_index = questions_set[context_col_name].astype(str).apply(len).idxmax()
    # Extract the context from that row
    longest_context = questions_set.loc[max_char_index, context_col_name]
    length_longest_context_tokenized = len(tokenizer.encode(longest_context))
    length_longest_context_tokenized_with_margin = int(length_longest_context_tokenized * (1 + margin))
    print("Longest context is ", length_longest_context_tokenized, " tokens. With Safety margin, that sets the context size to " , length_longest_context_tokenized_with_margin)
    
    return length_longest_context_tokenized_with_margin

def export_results(
    results_dataframe: pd.DataFrame,
    args: Dict[str, Any],
    timestamp: str,
    model_name: str,
    output_file_name: str
) -> None:
    """
    Exports the results DataFrame to a CSV file and saves the configuration to a JSON file.

    Args:
        results_dataframe (pd.DataFrame): The DataFrame containing the results to export.
        args (Dict[str, Any]): The script's command-line arguments.
        timestamp (str): The timestamp for the current run.
        model_name (str): The full name of the model used.
        output_file_name (str): The path to the output CSV file.
    """
    results_dataframe.to_csv(output_file_name, index=False)
    config = {
        "timestamp": timestamp,
        "test_mode": args['test_mode'],
        "questionnaire": args['questionnaire'],
        "model_name": model_name,
        "model_type": "Base" if 'base' in args['model'] else "Chat",
        "prompt_context": args['prompt_context'],
        "context_column_name": args['context_column'],
        "persona": args['persona'],
        "number_of_choice_permutations": args['number_permutations'],
        "permutation_strategy": args['permutation_strategy'],
        "use_vllm": args['use_vllm'],
        "gpu_memory_utilization": args['gpu_memory_utilization'] if args['use_vllm'] else None,
        "hf_batch_size": args['batch_size'] if not args['use_vllm'] else None,
        "output_file_name": output_file_name,
    }
    with open(output_file_name.replace('results.csv', 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
def main() -> None:
    """
    Main function to run the questionnaire answering and uncertainty analysis pipeline.
    """

    print("---------------Setting Up---------------")

    args = parse_args()
    model_name_short = args['model']
    model_name = MODEL_NAME_DICT[model_name_short]
    root_dir = "../"
    # If the timestamp already exists, wait a second and try again so we don't overwrite previous results
    while True:
        timestamp = pd.Timestamp.now().strftime("%Y_%m_%d_%H_%M_%S")
        if not os.path.exists(root_dir + "results/" + model_name.split("/")[1] + "/" + timestamp):
            break
        time.sleep(1)

    output_file_name = root_dir + "results/" + model_name.split("/")[1] + "/" + timestamp + "/results.csv"

    persona = None
    personas_file = 'personas_base.json' if 'base' in model_name_short else 'personas_chat.json'
    with open(root_dir + 'steering_techniques/prompting/' + personas_file, 'r') as f:
        personas = json.load(f)
        persona = personas[args['persona']]['persona']

    print("Running with specs:")
    print("Timestamp: ", timestamp)
    print("Test mode: ", args['test_mode'])
    print("Questionnaire: ", args['questionnaire'])
    print("Model: ", model_name)
    print("Model Type: ", "Base" if 'base' in model_name_short else "Chat/Instruction-Tuned")
    print("Persona: ", persona)
    print("Prompt context: ", args['prompt_context'])
    print("Context column name: ", args['context_column'])
    print("Number of choice permutations: ", args['number_permutations'])
    print("Output file name: ", output_file_name)
    print("Permutation strategy: ", args['permutation_strategy'])
    print("Using Engine: ", "vLLM" if args['use_vllm'] else "HF Transformers")
    if args['use_vllm']:
        print("GPU VRAM Utilisation: ", args['gpu_memory_utilization'])
    elif not args['use_vllm']:
        print("Batch size: ", args['batch_size'])
    
    # Make sure the results directory exists
    os.makedirs(root_dir + 'results/' + model_name.split("/")[1] + '/' + timestamp, exist_ok=True)
    
    print("------------Preparing Questionnaire------------")
    if args['questionnaire'] == 'political_compass':
        questionnaire_file_path = root_dir + 'questionnaires/political_compass.csv'
    
    questions_set = pd.read_csv(questionnaire_file_path)

    if args['test_mode']:
        questions_set = questions_set.head(5)

    print("-------------Initialising LLM-------------")
    # We get the config to set the vocab size
    model, tokenizer = None, None
    if not args['use_vllm']:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()

    else:
        # Using vLLM
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
        vocab_size = len(tokenizer)
        context_size_requirement = get_context_size_requirement(questions_set, tokenizer, args['context_column'])

        model = LLM(
            model=model_name,
            task="generate",
            trust_remote_code=True,
            max_logprobs=vocab_size,
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=args['gpu_memory_utilization'],
            max_model_len=context_size_requirement,
        )

    print("-------------Doing Inference-------------")
    questions_set_with_probs = generate_uncertainty_for_questionnaire(
        model, tokenizer, questions_set, args, persona=persona)

    print("-------------Saving Results-------------")
    export_results(questions_set_with_probs, args, timestamp, model_name, output_file_name)

    if args['use_vllm']:
        print("Gracefully turning off vLLM Engine...")
        model.llm_engine.engine_core.shutdown()

    print("------------------Done!-----------------")

if __name__ == "__main__":
    main()
