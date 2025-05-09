import os
import json
import datetime
import time
import glob
from anthropic import Anthropic, APIError, RateLimitError, APITimeoutError
import tiktoken


# --- Configuration ---
ANTHROPIC_API_KEY = "YOUR KEY GOES HERE" # Replace with your actual key

ANTHROPIC_MODEL_NAME = "claude-3-5-sonnet-20240620"

SYSTEM_PROMPT_CHUNK_INSIGHTS = """You are an expert AI assistant. Your task is to analyze a new chunk of Telegram chat messages.
You have been provided with a 'Previous Conversation Summary' which summarizes key points from messages processed before this current chunk.
Based on BOTH the 'Previous Conversation Summary' AND the 'Current Message Chunk', identify and extract NEW key insights, decisions, important questions, action items, and main topics from the 'Current Message Chunk' ONLY.
Do not repeat information already well-covered in the 'Previous Conversation Summary' unless the new chunk significantly updates or refutes it or adds crucial new details.
If the 'Current Message Chunk' contains no new salient information, or is just noise/greetings/banter without substance when considering the previous summary, respond with the exact phrase "NO_NEW_INSIGHTS".
Otherwise, present the new insights for the current chunk concisely and clearly, using bullet points for distinct items. Focus on actionable and informative content.
The chat messages are in the format 'Sender Name: Message content' or 'Sender Name (replying to 'OriginalSender: Snippet'): Message content'.
"""

SYSTEM_PROMPT_COMPACT_INSIGHTS_COLLECTION = """You are an expert AI assistant. You will be given a 'Collection of Previous Insights' from a long-running Telegram chat. These insights were generated sequentially from chunks of the conversation.
Your task is to synthesize and compact this collection into a single, coherent, and concise 'Updated Conversation Summary so far'.
This summary must retain all unique key information, decisions, questions, action items, and important topic developments mentioned in the individual insights.
Eliminate redundancy, combine related points, and maintain chronological flow where sensible. The goal is a dense, informative summary that is shorter than the sum of its parts but captures all critical information needed to understand the conversation's progression.
Focus on factual extraction and key developments. Use clear language and bullet points for distinct pieces of information where appropriate. This summary will be used as context for processing future message chunks OR as input to a final report.
"""

SYSTEM_PROMPT_FINAL_REPORT = """You are an expert AI assistant. You have been provided with a 'Consolidated Summary of Key Conversation Insights', which covers a long Telegram chat. These insights have been chronologically ordered and potentially pre-summarized.
Your task is to synthesize these insights into a single, well-structured, highly readable, and comprehensive 'Final Report' of the entire conversation.

The report should be formatted for easy readability by a human. Use clear headings, paragraphs, and MOST IMPORTANTLY, extensive use of bullet points for:
- Key Decisions Made (with when/who if identifiable)
- Major Action Items (assigned to whom if identifiable, and any deadlines)
- Significant Questions Raised (and their resolutions or if they remain open)
- Critical Takeaways and Conclusions

Structure your report logically:
1.  **Executive Summary:** A brief overview (2-3 paragraphs) of the entire conversation's purpose, main outcomes, and critical conclusions.
2.  **Main Themes and Discussion Flow:** Describe the overarching topics and how the conversation evolved.
3.  **Detailed Breakdown:**
    * **Key Decisions:** Use bullet points. For each decision, briefly state the decision and any context.
    * **Action Items:** Use bullet points. For each action item, specify the task, who is responsible (if known), and any deadlines.
    * **Important Questions & Resolutions:** Use bullet points. List key questions, and their answers or current status.
    * **Other Significant Insights/Events:** Use bullet points for any other notable information, topic shifts, or unresolved issues.
4.  **Overall Conclusion/Next Steps (if apparent):** Summarize the final state of the discussion.

Ensure the report is detailed enough to be a standalone useful document. Do not just list the input insights; synthesize, structure, and rephrase them into this formal report format. Be comprehensive.
"""

MESSAGES_PER_CHUNK = 50
MAX_PREVIOUS_INSIGHTS_TOKENS_FOR_CHUNK_PROCESSING = 100000
MAX_TOKENS_FOR_INSIGHT_COLLECTION_COMPACTION = 150000
MAX_INSIGHTS_PER_META_CHUNK_FOR_REPORT = 30
MAX_TOKENS_FOR_FINAL_REPORT_GENERATION = 180000
MAX_API_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 5
MAX_BACKOFF_SECONDS = 120

try:
    TOKENIZER = tiktoken.encoding_for_model(ANTHROPIC_MODEL_NAME)
except Exception:
    TOKENIZER = tiktoken.get_encoding("cl100k_base")

def count_tokens(text_string):
    if not text_string: return 0
    return len(TOKENIZER.encode(str(text_string)))

def get_anthropic_client():
    if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY.startswith("sk-ant-api03-xxx"):
        print("CRITICAL Error: ANTHROPIC_API_KEY is not set or is still the placeholder value.")
        raise ValueError("ANTHROPIC_API_KEY not configured.")
    return Anthropic(api_key=ANTHROPIC_API_KEY)

def load_state_data(state_file_path):
    if os.path.exists(state_file_path):
        try:
            with open(state_file_path, 'r', encoding='utf-8') as f: data = json.load(f)
            data.setdefault("last_processed_message_id", 0)
            data.setdefault("last_processed_message_timestamp_unix", 0)
            data.setdefault("individual_useful_insights", [])
            data.setdefault("latest_report_filename", None)
            data.setdefault("report_generated_at_utc", None)
            data.setdefault("system_prompts_used", {})
            return data
        except Exception as e: print(f"Warning: Error loading {state_file_path}: {e}. Starting fresh.")
    return {"last_processed_message_id": 0, "last_processed_message_timestamp_unix": 0,
            "individual_useful_insights": [], "latest_report_filename": None,
            "report_generated_at_utc": None, "system_prompts_used": {}}

def save_state_data(state_file_path, data):
    try:
        data["system_prompts_used"]["chunk_insights"] = SYSTEM_PROMPT_CHUNK_INSIGHTS
        data["system_prompts_used"]["compact_insights_collection"] = SYSTEM_PROMPT_COMPACT_INSIGHTS_COLLECTION
        data["system_prompts_used"]["final_report"] = SYSTEM_PROMPT_FINAL_REPORT
        with open(state_file_path, 'w', encoding='utf-8') as f: json.dump(data, f, indent=4,ensure_ascii=False)
    except Exception as e: print(f"CRITICAL Error saving state data to {state_file_path}: {e}")

def find_latest_export_file(chat_folder_path, state_file_name):
    json_files = glob.glob(os.path.join(chat_folder_path, "*.json"))
    export_files = [f for f in json_files if os.path.basename(f) != state_file_name]
    if not export_files: return None
    return max(export_files, key=os.path.getmtime)

def parse_telegram_message_text(text_data):
    if isinstance(text_data, str): return text_data.strip()
    if isinstance(text_data, list):
        return "".join(
            segment['text'] if isinstance(segment, dict) and 'text' in segment else (segment if isinstance(segment, str) else '')
            for segment in text_data).strip()
    return ""

def load_and_format_new_messages(export_file_path, last_id_processed, last_timestamp_unix_processed):
    try:
        with open(export_file_path, 'r', encoding='utf-8') as f: data = json.load(f)
    except Exception as e: print(f"Error reading/parsing export file {export_file_path}: {e}"); return [],0,0
    all_messages_in_file = data.get('messages', []);
    if not all_messages_in_file: return [],0,0
    message_lookup_dict, valid_messages_for_sorting = {},[]
    for msg_data in all_messages_in_file:
        try:
            msg_id,msg_timestamp_unix = int(msg_data.get('id')),int(msg_data.get('date_unixtime'))
            msg_data['id'],msg_data['date_unixtime'] = msg_id,msg_timestamp_unix
            valid_messages_for_sorting.append(msg_data)
            if msg_data.get('type') == 'message': message_lookup_dict[msg_id] = msg_data
        except (ValueError,TypeError): pass
    if not valid_messages_for_sorting: return [],0,0
    valid_messages_for_sorting.sort(key=lambda m: (m['date_unixtime'],m['id']))
    
    # Ensure there are messages before trying to access indices
    if not valid_messages_for_sorting:
        return [], 0, 0
        
    first_id_in_file,last_id_in_file = valid_messages_for_sorting[0]['id'],valid_messages_for_sorting[-1]['id']
    new_formatted_messages = []
    for msg_data in valid_messages_for_sorting:
        msg_id,msg_timestamp_unix = msg_data['id'],msg_data['date_unixtime']
        if msg_timestamp_unix < last_timestamp_unix_processed or \
           (msg_timestamp_unix == last_timestamp_unix_processed and msg_id <= last_id_processed): continue
        if msg_data.get('type') != 'message': continue
        text_content,sender_name = parse_telegram_message_text(msg_data.get('text','')),msg_data.get('from','Unknown Sender') or 'System'
        llm_text_prefix,reply_to_id = f"From {sender_name}:",msg_data.get('reply_to_message_id')
        
        # ***** CORRECTED SECTION START *****
        if reply_to_id and reply_to_id in message_lookup_dict:
            original_msg = message_lookup_dict[reply_to_id] # Assign original_msg first
            original_sender_name = original_msg.get('from','Original Sender') or 'System' # Then use it
            original_text = parse_telegram_message_text(original_msg.get('text',''))
            if not original_text:
                if original_msg.get('file'): original_text = f"[File: {original_msg.get('file_name','attachment')}]"
                elif original_msg.get('photo'): original_text = "[Photo]"
                elif original_msg.get('sticker'): original_text = f"[Sticker {original_msg.get('sticker_emoji','')}]".strip()
            if original_text:
                reply_snippet = (original_text[:75] + '...') if len(original_text) > 75 else original_text
                llm_text_prefix = f"From {sender_name} (replying to '{original_sender_name}: {reply_snippet}'):"
        # ***** CORRECTED SECTION END *****

        final_text_for_llm = ""
        if text_content: final_text_for_llm = f"{llm_text_prefix} {text_content}"
        elif msg_data.get('file'): final_text_for_llm = f"{llm_text_prefix} [Sent a file: {msg_data.get('file_name','attachment')}]"
        elif msg_data.get('photo'): final_text_for_llm = f"{llm_text_prefix} [Sent a photo]"
        elif msg_data.get('sticker'): final_text_for_llm = f"{llm_text_prefix} [Sent a sticker {msg_data.get('sticker_emoji','')}]".strip()
        else: continue
        new_formatted_messages.append({"id":msg_id,"text_for_llm":final_text_for_llm,"timestamp_unix":msg_timestamp_unix,"timestamp_utc":msg_data.get('date','')})
    return new_formatted_messages,first_id_in_file,last_id_in_file

def call_anthropic_api(client, system_prompt, user_messages_content, max_response_tokens=2048, attempt=1):
    full_prompt_messages = [{"role": "user", "content": user_messages_content}] if isinstance(user_messages_content, str) else user_messages_content
    try:
        # print(f"API call attempt {attempt}/{MAX_API_RETRIES} to model {ANTHROPIC_MODEL_NAME} with ~{count_tokens(system_prompt) + count_tokens(user_messages_content)} input tokens.")
        response = client.messages.create(model=ANTHROPIC_MODEL_NAME, max_tokens=max_response_tokens, system=system_prompt, messages=full_prompt_messages)
        return response.content[0].text.strip()
    except (APIError, APITimeoutError) as e:
        error_code = e.status_code if hasattr(e, 'status_code') else 'Timeout'
        error_message = e.message if hasattr(e, 'message') and e.message else str(e)
        error_prefix = f"ANTHROPIC_API_ERROR (Code {error_code}): {error_message}"
        if error_code == 529 or error_code == 429 or (isinstance(error_code, int) and 500 <= error_code <= 599) or error_code == 'Timeout':
            if attempt < MAX_API_RETRIES:
                backoff = min(INITIAL_BACKOFF_SECONDS * (2**(attempt - 1)), MAX_BACKOFF_SECONDS) * (1 + os.urandom(1)[0]/255.0 * 0.2) # jitter
                print(f"{error_prefix}. Retrying in {backoff:.2f}s (attempt {attempt+1}/{MAX_API_RETRIES})...")
                time.sleep(backoff)
                return call_anthropic_api(client, system_prompt, user_messages_content, max_response_tokens, attempt + 1)
            else: print(f"{error_prefix}. Max retries reached."); return error_prefix
        else: print(f"Non-retriable {error_prefix}."); return error_prefix
    except Exception as e: print(f"Unexpected error calling Anthropic API: {e}"); return f"UNEXPECTED_ANTHROPIC_API_ERROR: {str(e)}"

def compact_insights_collection(client, insights_texts_list, max_tokens_for_context, prompt_for_compaction):
    if not insights_texts_list: return ""
    full_insights_str = "\n\n---\n\n".join(insights_texts_list)
    current_tokens = count_tokens(full_insights_str)

    if current_tokens > max_tokens_for_context:
        print(f"Insights collection ({current_tokens} tokens) exceeds limit ({max_tokens_for_context}). Compacting...")
        compaction_input_str = full_insights_str
        max_allowed_input_for_compactor = MAX_TOKENS_FOR_INSIGHT_COLLECTION_COMPACTION - count_tokens(prompt_for_compaction) - 2000 
        if count_tokens(compaction_input_str) > max_allowed_input_for_compactor:
            print(f"Warning: Full insights string for compaction is very large. Truncating input for compaction API call.")
            encoded_tokens = TOKENIZER.encode(compaction_input_str)
            truncated_tokens = encoded_tokens[:max_allowed_input_for_compactor]
            compaction_input_str = TOKENIZER.decode(truncated_tokens)
            print(f"Truncated compaction input to approx {count_tokens(compaction_input_str)} tokens.")
        
        user_prompt = f"Please compact the following collection of insights according to the instructions:\n\n{compaction_input_str}"
        compacted_summary = call_anthropic_api(client, prompt_for_compaction, user_prompt, max_response_tokens=3000) 
        
        if compacted_summary.startswith("ANTHROPIC_") or compacted_summary.startswith("UNEXPECTED_"):
            print(f"Error during insights compaction: {compacted_summary}. Using uncompacted (but potentially truncated for safety) insights.")
            safe_str = ""; temp_tokens = 0
            for text in reversed(insights_texts_list): 
                text_tokens = count_tokens(text) + 5
                if temp_tokens + text_tokens <= max_tokens_for_context:
                    safe_str = text + "\n\n---\n\n" + safe_str; temp_tokens += text_tokens
                else: break
            return safe_str.strip()
        else:
            print(f"Compaction successful. New token count for insights context: {count_tokens(compacted_summary)}")
            return compacted_summary
    return full_insights_str

def generate_and_save_final_report_txt(client, state_data_ref, state_file_path_ref, chat_folder_path, new_insights_were_added_this_run=True):
    should_generate_report = new_insights_were_added_this_run or \
                             (state_data_ref["individual_useful_insights"] and \
                              (not state_data_ref.get("latest_report_filename") or \
                               state_data_ref.get("report_generated_at_utc") is None or 
                               not os.path.exists(os.path.join(chat_folder_path, state_data_ref.get("latest_report_filename","----.txt")))))

    if not should_generate_report:
        print("Report generation not triggered: No new insights and a valid report likely exists.")
        return True 

    print("Preparing to generate/regenerate final report...")
    all_individual_insights = [item["insight_text"] for item in state_data_ref["individual_useful_insights"]
                               if item["insight_text"] and item["insight_text"] != "NO_NEW_INSIGHTS"]

    if not all_individual_insights:
        print("No substantial insights available to generate a final report.")
        state_data_ref['latest_report_filename'] = None 
        state_data_ref['report_generated_at_utc'] = datetime.datetime.utcnow().isoformat() + "Z"
        save_state_data(state_file_path_ref, state_data_ref)
        return True

    current_input_for_report = "\n\n---\n\n".join(all_individual_insights)
    current_tokens = count_tokens(current_input_for_report)
    print(f"Total tokens from {len(all_individual_insights)} individual insights for final report (before meta-compaction): {current_tokens}")

    if current_tokens > MAX_TOKENS_FOR_FINAL_REPORT_GENERATION:
        print(f"Combined insights ({current_tokens} tokens) exceed final report generation limit ({MAX_TOKENS_FOR_FINAL_REPORT_GENERATION}). Performing multi-stage summarization for report input.")
        meta_chunked_insights = [all_individual_insights[i:i + MAX_INSIGHTS_PER_META_CHUNK_FOR_REPORT]
                                 for i in range(0, len(all_individual_insights), MAX_INSIGHTS_PER_META_CHUNK_FOR_REPORT)]
        summarized_meta_chunks = []
        print(f"Splitting {len(all_individual_insights)} insights into {len(meta_chunked_insights)} meta-chunks for pre-report summarization.")
        for idx, meta_chunk in enumerate(meta_chunked_insights):
            print(f"Summarizing meta-chunk {idx+1}/{len(meta_chunked_insights)}...")
            meta_chunk_text = "\n\n---\n\n".join(meta_chunk)
            meta_summary = call_anthropic_api(client, SYSTEM_PROMPT_COMPACT_INSIGHTS_COLLECTION,
                                              f"Please provide a concise summary of the following insights, which are part of a larger collection for a final report:\n\n{meta_chunk_text}",
                                              max_response_tokens=2000)
            if meta_summary.startswith("ANTHROPIC_") or meta_summary.startswith("UNEXPECTED_"):
                print(f"Error summarizing meta-chunk {idx+1}: {meta_summary}. This part may be missing from the final report.")
                summarized_meta_chunks.append(f"[Error summarizing meta-chunk {idx+1}: {meta_summary}]")
            else:
                summarized_meta_chunks.append(meta_summary)
                print(f"Meta-chunk {idx+1} summarized. Length: {len(meta_summary)}, Tokens: {count_tokens(meta_summary)}")
        current_input_for_report = "\n\n---\n\n".join(summarized_meta_chunks)
        final_input_tokens = count_tokens(current_input_for_report)
        print(f"Final input for report generation after meta-summarization has {final_input_tokens} tokens.")
        if final_input_tokens > MAX_TOKENS_FOR_FINAL_REPORT_GENERATION:
             print(f"WARNING: Even after meta-summarization, the input for final report ({final_input_tokens} tokens) is large. Report might be truncated by API or less coherent.")

    print("Calling API for final report generation...")
    final_report_text_content = call_anthropic_api(
        client, SYSTEM_PROMPT_FINAL_REPORT,
        f"Using the following consolidated key insights, please generate the final report as per the detailed instructions:\n\n{current_input_for_report}",
        max_response_tokens=4000)

    if final_report_text_content.startswith("ANTHROPIC_") or final_report_text_content.startswith("UNEXPECTED_"):
        print(f"CRITICAL Error generating final report text: {final_report_text_content}")
        state_data_ref['latest_report_filename'] = None 
        state_data_ref['report_generated_at_utc'] = datetime.datetime.utcnow().isoformat() + "Z"
        save_state_data(state_file_path_ref, state_data_ref)
        return False

    report_date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    report_filename = f"{os.path.basename(chat_folder_path)}-Report-{report_date_str}.txt"
    report_file_path = os.path.join(chat_folder_path, report_filename)
    try:
        with open(report_file_path, 'w', encoding='utf-8') as f:
            f.write(f"Final Report for: {os.path.basename(chat_folder_path)}\n")
            f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}\n")
            f.write("=" * 30 + "\n\n")
            f.write(final_report_text_content)
        print(f"Final report successfully saved to: {report_file_path}")
        state_data_ref['latest_report_filename'] = report_filename
        state_data_ref['report_generated_at_utc'] = datetime.datetime.utcnow().isoformat() + "Z"
        save_state_data(state_file_path_ref, state_data_ref)
        return True
    except Exception as e:
        print(f"CRITICAL Error saving final report to .txt file: {e}")
        state_data_ref['latest_report_filename'] = None
        state_data_ref['report_generated_at_utc'] = datetime.datetime.utcnow().isoformat() + "Z"
        save_state_data(state_file_path_ref, state_data_ref)
        return False

def process_chat_folder(chat_folder_name):
    print(f"\n=== Processing Chat Folder: {chat_folder_name} ===")
    chat_folder_path = os.path.abspath(chat_folder_name)
    if not os.path.isdir(chat_folder_path): print(f"Error: Directory '{chat_folder_path}' not found."); return

    state_file_name = f"{os.path.basename(chat_folder_path)}-state.json"
    state_file_path = os.path.join(chat_folder_path, state_file_name)
    state_data = load_state_data(state_file_path)
    
    print(f"Loaded state. Last processed message ID: {state_data['last_processed_message_id']}, Timestamp: {datetime.datetime.fromtimestamp(state_data['last_processed_message_timestamp_unix']).strftime('%Y-%m-%d %H:%M:%S') if state_data['last_processed_message_timestamp_unix'] > 0 else 'N/A'}")
    if state_data.get('latest_report_filename') and os.path.exists(os.path.join(chat_folder_path, state_data.get('latest_report_filename','----.txt'))): # Check if file actually exists
        print(f"Last report generated: {state_data['latest_report_filename']} at {state_data.get('report_generated_at_utc', 'N/A')}")

    try: anthropic_client = get_anthropic_client()
    except ValueError: return

    export_file = find_latest_export_file(chat_folder_path, state_file_name)
    
    if not export_file:
        print("No new export files found to process.")
        generate_and_save_final_report_txt(anthropic_client, state_data, state_file_path, chat_folder_path, new_insights_were_added_this_run=False)
        return

    print(f"Processing export file: {export_file}")
    new_messages, first_id_in_file, last_id_in_file = load_and_format_new_messages(export_file, state_data['last_processed_message_id'], state_data['last_processed_message_timestamp_unix'])

    if not new_messages:
        print("No new messages to process in the export file after filtering.")
        if last_id_in_file > 0 and state_data['last_processed_message_id'] >= last_id_in_file :
            print(f"Export file '{os.path.basename(export_file)}' fully processed or older. Deleting.")
            try: os.remove(export_file); print(f"Deleted: {export_file}")
            except OSError as e: print(f"Error deleting old export file: {e}")
        generate_and_save_final_report_txt(anthropic_client, state_data, state_file_path, chat_folder_path, new_insights_were_added_this_run=False)
        return

    print(f"Found {len(new_messages)} new messages to process.")

    previous_insights_context_str = compact_insights_collection(
        anthropic_client,
        [item["insight_text"] for item in state_data["individual_useful_insights"]],
        MAX_PREVIOUS_INSIGHTS_TOKENS_FOR_CHUNK_PROCESSING,
        SYSTEM_PROMPT_COMPACT_INSIGHTS_COLLECTION
    )
    if previous_insights_context_str.startswith("ANTHROPIC_") or previous_insights_context_str.startswith("UNEXPECTED_"):
        print(f"Failed to prepare initial insights context due to API error: {previous_insights_context_str}. Aborting run."); return

    num_chunks = (len(new_messages) + MESSAGES_PER_CHUNK - 1) // MESSAGES_PER_CHUNK
    any_new_insights_this_run, all_chunks_successful = False, True

    for i in range(num_chunks):
        chunk_messages = new_messages[i * MESSAGES_PER_CHUNK : (i + 1) * MESSAGES_PER_CHUNK]
        if not chunk_messages: continue
        print(f"\nProcessing Chunk {i+1}/{num_chunks} (Messages {chunk_messages[0]['id']} to {chunk_messages[-1]['id']})...")
        current_chunk_text = "\n".join([msg['text_for_llm'] for msg in chunk_messages])
        prompt_content_for_chunk = f"Previous Conversation Summary:\n{previous_insights_context_str if previous_insights_context_str else 'No previous summary available.'}\n\n---\n\nCurrent Message Chunk:\n{current_chunk_text}"
        
        chunk_insight_text = call_anthropic_api(anthropic_client, SYSTEM_PROMPT_CHUNK_INSIGHTS, prompt_content_for_chunk, max_response_tokens=1500)

        if chunk_insight_text.startswith("ANTHROPIC_") or chunk_insight_text.startswith("UNEXPECTED_"):
            print(f"CRITICAL Error generating insight for chunk {i+1}: {chunk_insight_text}"); all_chunks_successful = False; break

        if chunk_insight_text == "NO_NEW_INSIGHTS": print("LLM indicated no new salient insights for this chunk.")
        else:
            print(f"Insight for chunk {i+1}: {chunk_insight_text[:150]}..."); any_new_insights_this_run = True
            new_insight_obj = {"chunk_start_message_id": chunk_messages[0]["id"], "chunk_end_message_id": chunk_messages[-1]["id"], 
                               "chunk_start_message_timestamp_utc": chunk_messages[0]["timestamp_utc"], "chunk_end_message_timestamp_utc": chunk_messages[-1]["timestamp_utc"], 
                               "insight_text": chunk_insight_text, "generated_at_utc": datetime.datetime.utcnow().isoformat() + "Z"}
            state_data["individual_useful_insights"].append(new_insight_obj)
            
            previous_insights_context_str = compact_insights_collection(
                anthropic_client,
                [item["insight_text"] for item in state_data["individual_useful_insights"]],
                MAX_PREVIOUS_INSIGHTS_TOKENS_FOR_CHUNK_PROCESSING,
                SYSTEM_PROMPT_COMPACT_INSIGHTS_COLLECTION
            )
            if previous_insights_context_str.startswith("ANTHROPIC_") or previous_insights_context_str.startswith("UNEXPECTED_"):
                print(f"Failed to update/re-compact previous_insights_context after new insight: {previous_insights_context_str}. Subsequent chunks may have stale/failed context."); all_chunks_successful = False; break
        
        state_data['last_processed_message_id'] = chunk_messages[-1]['id']
        state_data['last_processed_message_timestamp_unix'] = chunk_messages[-1]['timestamp_unix']
        state_data['latest_report_filename'], state_data['report_generated_at_utc'] = None, None 
        save_state_data(state_file_path, state_data)
        print(f"Progress saved. Last processed ID: {state_data['last_processed_message_id']}")

    report_generation_successful_this_run = False
    if all_chunks_successful and new_messages: 
        print("\nAll new message chunks from export file processed.")
        report_generation_successful_this_run = generate_and_save_final_report_txt(anthropic_client, state_data, state_file_path, chat_folder_path, any_new_insights_this_run)
        
        if report_generation_successful_this_run: 
            print(f"Attempting to delete processed export file: {export_file}")
            try: os.remove(export_file); print(f"Deleted export file: {export_file}")
            except OSError as e: print(f"Error deleting export file {export_file}: {e}")
        else:
            print(f"\nFinal report generation was not successful. The export file '{os.path.basename(export_file)}' will NOT be deleted.")
            
    elif not new_messages and export_file : 
        print(f"Export file '{os.path.basename(export_file)}' contained no new messages to process for insights.")
        generate_and_save_final_report_txt(anthropic_client, state_data, state_file_path, chat_folder_path, new_insights_were_added_this_run=False)

    elif not all_chunks_successful: 
        print(f"\nProcessing of individual message chunks in '{os.path.basename(export_file)}' was not fully successful. The file will NOT be deleted.")
        print("Progress up to the last successful chunk (if any) has been saved. Please check errors and re-run.")

    print(f"=== Finished processing Chat Folder: {chat_folder_name} ===")

if __name__ == "__main__":
    target_folder_name_input = input("Enter the name of the chat folder to process (e.g., 'nockchain_chat'): ")
    if target_folder_name_input:
        process_chat_folder(target_folder_name_input)
    else:
        print("No target folder name provided. Exiting.")
