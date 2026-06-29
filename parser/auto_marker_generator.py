import torch
import re
import numpy as np
from transformers import AutoTokenizer, RobertaForMaskedLM
import logging
import random
from collections import defaultdict
import time
logging.getLogger("transformers").setLevel(logging.ERROR)
PATH_PATTERN = re.compile(r'(?:[a-zA-Z0-9]+:/{1,3}|/{2,3}|/|[a-zA-Z0-9._/+-]+/)(?:(?:\(\.\*\?\))|[^\s),])*(?:(?:\(\.\*\?\))|[^\s),.])')

class UnsupervisedMarkerExtractor:
    def __init__(self, model_path='roberta-base'):
        self.total_time = 0.0
        self.device = self._get_optimal_device()
        print(f"[MarkerExtractor] Loading RoBERTa Model on {self.device}...", flush=True)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            self.model = RobertaForMaskedLM.from_pretrained(model_path).to(self.device)
            self.model.eval()
            if not self.tokenizer.is_fast:
                print("⚠️ 警告: 加载的是 Slow Tokenizer，offset_mapping 可能无法工作！", flush=True)
            else:
                print("[MarkerExtractor] Fast Tokenizer loaded successfully.", flush=True)
        except Exception as e:
            print(f"[MarkerExtractor] Error loading model: {e}", flush=True)
            raise e
    def _is_valid_static_ed_word(self, token, log_content):

        lower_token = token.lower().strip('.,')
        if not lower_token.endswith('ed'):
            return False
        if '_' in token or '-' in token:
            return False
        if token.isupper():
            return False
        if not (4 < len(lower_token) < 15):
            return False

        t_idx = log_content.find(token)
        if t_idx != -1:
            pre_text = log_content[:t_idx]
            post_text = log_content[t_idx + len(token):]


            after_token_text = post_text.lstrip()
            if after_token_text.startswith('-'):
                return False
            words_after = post_text.strip().split()
            if len(words_after) > 1 and words_after[0].lower() in ['from', 'at', 'on', 'port']:
                second_w = words_after[1].strip('.,;:')
                if re.match(r'^(\d{1,3}\.){3}\d{1,3}', second_w) or second_w.isdigit():
                    return False

            pairs = [('[', ']'), ("'", "'"), ('"', '"'), ('(', ')'), ('{', '}')]
            for open_b, close_b in pairs:
                last_open = pre_text.rfind(open_b)
                if last_open != -1:
                    if close_b not in pre_text[last_open:]:
                        if post_text.find(close_b) != -1:
                            return False


            last_colon = pre_text.rfind(':')
            if last_colon != -1:
                if len(re.findall(r'\w+', pre_text[last_colon:])) <= 5:
                    return False
            last_eq = pre_text.rfind('=')
            if last_eq != -1:
                if len(re.findall(r'\w+', pre_text[last_eq:])) <= 3:
                    return False

        return True

    def _is_valid_static_ty_word(self, token, log_content):

        lower_token = token.lower().strip('.,')
        if not lower_token.endswith('ty'):
            return False
        if '_' in token or '-' in token:
            return False
        if token.isupper():
            return False
        if not (4 < len(lower_token) < 15):
            return False

        t_idx = log_content.find(token)
        if t_idx != -1:
            pre_text = log_content[:t_idx]
            post_text = log_content[t_idx + len(token):]


            after_token_text = post_text.lstrip()
            if after_token_text.startswith('-'):
                return False


            pairs = [('[', ']'), ("'", "'"), ('"', '"'), ('(', ')'), ('{', '}')]
            for open_b, close_b in pairs:
                last_open = pre_text.rfind(open_b)
                if last_open != -1:
                    if close_b not in pre_text[last_open:]:
                        if post_text.find(close_b) != -1:
                            return False


            last_colon = pre_text.rfind(':')
            if last_colon != -1:
                if len(re.findall(r'\w+', pre_text[last_colon:])) <= 5:
                    return False
            last_eq = pre_text.rfind('=')
            if last_eq != -1:
                if len(re.findall(r'\w+', pre_text[last_eq:])) <= 3:
                    return False

        return True


    def _is_inside_path(self, text, start_pos, end_pos):
        for match in PATH_PATTERN.finditer(text):
            path_str = match.group(0)

            if ':' in path_str or path_str.startswith('//') or path_str.count('/') >= 2:
                if match.start() <= start_pos and match.end() >= end_pos:
                    return True
        return False

    def _get_context_signature(self, log_text, split_token):


        temp_log = log_text.replace(split_token, " ")


        pure_alpha_tokens = re.findall(r'\b[a-zA-Z]{2,}\b', temp_log)

        return " ".join(pure_alpha_tokens)

    def detect_binary_split_with_bert(self, log_group, bert_threshold=0.1):
        start_t = time.time()
        try:
            return self.detect_binary_split_with_bert_core(log_group, bert_threshold)
        finally:
            self.total_time += (time.time() - start_t)
    def detect_binary_split_with_bert_core(self, log_group, bert_threshold=0.1):

        if not log_group or len(log_group) < 2:
            return None
        dc_groups = {}
        for log in log_group:

            match = re.search(r'([a-zA-Z_]+)::([a-zA-Z_]+)', log)
            found_dc_token = None

            if match:
                t2 = match.group(2)

                if self._is_valid_double_colon_static(t2, log):
                    found_dc_token = t2


            split_key = found_dc_token if found_dc_token else "<NO_DC>"
            if split_key not in dc_groups:
                dc_groups[split_key] = []
            dc_groups[split_key].append(log)


        valid_dc_keys = [k for k in dc_groups.keys() if k != "<NO_DC>"]

        if len(valid_dc_keys) >= 2:

            any_passed_bert = False
            for dc_key in valid_dc_keys:

                score = self._calculate_compound_score(dc_groups[dc_key][0], dc_key)


                if score > (bert_threshold * 0.001):
                    any_passed_bert = True
                    break

            if any_passed_bert:
                #print(f"[Split Priority] Detected semantic split by double colon tokens: {valid_dc_keys}")
                return dc_groups
        ed_groups = {}
        for log in log_group:
            all_words = re.findall(r'\b[a-zA-Z]+\b', log)
            found_ed_in_this_log = None
            for word in all_words:
                if self._is_valid_static_ed_word(word, log):
                    found_ed_in_this_log = word.strip('.')
                    break

            split_key = found_ed_in_this_log if found_ed_in_this_log else "<NO_ED>"
            if split_key not in ed_groups:
                ed_groups[split_key] = []
            ed_groups[split_key].append(log)


        valid_ed_keys = [k for k in ed_groups.keys() if k != "<NO_ED>"]

        if len(valid_ed_keys) >= 2:

            any_passed_bert = False
            for ed_key in valid_ed_keys:

                score = self._calculate_compound_score(ed_groups[ed_key][0], ed_key)

                if score > (bert_threshold * 0.001):
                    any_passed_bert = True
                    break

            if any_passed_bert:
                print(f"[Split Priority] Detected semantic split by -ed suffixes: {valid_ed_keys}")

                return ed_groups
        groups = {}

        token_pattern = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_\-\.]*')
        TEMPORAL_KEYWORDS = {
            'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
            'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun',
        }

        for log in log_group:
            first_token = "<UNKNOWN>"

            for match in token_pattern.finditer(log):

                if not self._is_inside_path(log, match.start(), match.end()):

                    t = match.group(0)
                    start, end = match.start(), match.end()
                    t = re.sub(r'\.{2,}\d+$', '...', t)
                    if '.' in t.strip('.'):
                        continue


                    if end < len(log) and log[end] == '.':
                        if end + 1 < len(log) and log[end + 1].isalnum():
                            continue
                    if t.lower() in TEMPORAL_KEYWORDS:
                        continue
                    if not (len(t) > 4 and re.fullmatch(r'[0-9a-fA-F]+', t)):
                        first_token = t
                        break
            if first_token not in groups:
                groups[first_token] = []
            groups[first_token].append(log)
        if len(groups) > 10:
            print(f"[Split Aborted] Too many unique first tokens ({len(groups)}). "
                f"This usually indicates the first token is a variable (like IDs).")
            return None
        # === Step 1.2 ===
        units_set = {'B', 'KB', 'MB', 'GB'}
        new_refined_groups_dict = {}
        split_occurred = False  # 初始化分裂标志位

        for group_key, group_logs in groups.items():
            if not group_logs:
                continue

            representative_log = group_logs[0]
            words = representative_log.split()


            if any(word.strip('.,()[]:;') in units_set for word in words):

                len_subgroups = defaultdict(list)
                for log in group_logs:
                    len_subgroups[len(log.split())].append(log)

                for length, logs_in_len in len_subgroups.items():
                    if len(logs_in_len) < 2:
                        new_refined_groups_dict[f"{group_key}_len_{length}"] = logs_in_len
                        continue

                    tokenized = [l.split() for l in logs_in_len]
                    num_cols = length


                    dynamic_cols = set()
                    for i in range(num_cols):
                        raw_col_tokens = [t[i] for t in tokenized]
                        clean_tokens = [re.sub(r'[(),:]', '', t[i]) for t in tokenized]
                        unique_clean = set(clean_tokens)
                        # 判定逻辑：全数字、变化超过3种、或包含单位的变化词
                        if all(re.match(r'^-?\d+(\.\d+)?$', c) for c in clean_tokens):
                            dynamic_cols.add(i)
                        elif len(unique_clean) > 3:
                            dynamic_cols.add(i)
                        elif any(c.upper() in units_set for c in clean_tokens) and len(unique_clean) > 1:
                            dynamic_cols.add(i)
                        elif any(raw.count('.') >= 2 or '@' in raw for raw in raw_col_tokens):
                            dynamic_cols.add(i)

                    candidate_split_cols = [i for i in range(num_cols) if
                                            i not in dynamic_cols and len(set([t[i] for t in tokenized])) > 1]

                    if not candidate_split_cols:
                        new_refined_groups_dict[f"{group_key}_static_{length}"] = logs_in_len
                        continue


                    split_idx = candidate_split_cols[0]
                    for l in logs_in_len:
                        val = l.split()[split_idx]

                        if val not in new_refined_groups_dict:
                            new_refined_groups_dict[val] = []
                        new_refined_groups_dict[val].append(l)

                    split_occurred = True  # 标记已发生结构化分裂
            else:
                new_refined_groups_dict[group_key] = group_logs


        if split_occurred:
            print(f"[Split Success] Structural split found: {list(new_refined_groups_dict.keys())}")
            return new_refined_groups_dict

        groups = new_refined_groups_dict
            # === Step 1.5 ===

        if len(groups) == 1:
            first_token_key = list(groups.keys())[0]
            if first_token_key != "<UNKNOWN>":
                representative_log = groups[first_token_key][0]
                # 查找 token 位置
                t_idx = representative_log.find(first_token_key)
                if t_idx != -1:
                    # 验证是否存在包裹结构：向前找 '['，向后找 ']'
                    has_left_bracket = '[' in representative_log[:t_idx]
                    has_right_bracket = ']' in representative_log[t_idx + len(first_token_key):]

                    if has_left_bracket and has_right_bracket:
                        group_logs = groups[first_token_key]
                        log_token_sequences = []
                        for log in group_logs:
                            # 定位到 [] 结束位置，寻找其后的英文字符序列
                            bracket_end_idx = log.find(']')
                            after_bracket_text = log[bracket_end_idx + 1:] if bracket_end_idx != -1 else ""
                            # 提取纯字母单词作为分裂候选 (长度>=2)
                            tokens_after = re.findall(r'\b[a-zA-Z]{2,}\b', after_bracket_text)
                            log_token_sequences.append(tokens_after)

                        # 2. 深度扫描分歧列
                        split_col_idx = -1
                        max_search_depth = 4

                        for col_idx in range(max_search_depth):
                            # 提取该列所有日志的 Token
                            tokens_at_this_col = [seq[col_idx] if len(seq) > col_idx else "<MISSING>" for seq in
                                                  log_token_sequences]
                            if len(set(tokens_at_this_col)) > 1:
                                split_col_idx = col_idx
                                break

                        # 3. 判定分裂有效性与校验
                        if split_col_idx != -1:
                            new_groups = {}
                            for idx, log in enumerate(group_logs):
                                seq = log_token_sequences[idx]
                                split_token = seq[split_col_idx] if len(seq) > split_col_idx else "<MISSING>"
                                if split_token not in new_groups:
                                    new_groups[split_token] = []
                                new_groups[split_token].append(log)

                            if len(new_groups) > 1:
                                should_split = True


                                if should_split:
                                    if len(new_groups) > 4:

                                        sorted_keys = sorted(new_groups.keys(), key=lambda k: len(new_groups[k]),
                                                             reverse=True)
                                        final_groups = {k: new_groups[k] for k in sorted_keys[:4]}


                                        other_logs = []
                                        for i in range(4, len(sorted_keys)):
                                            other_logs.extend(new_groups[sorted_keys[i]])
                                        if other_logs:
                                            final_groups["<OTHER_BRANCHES>"] = other_logs
                                        groups = final_groups
                                    else:
                                        groups = new_groups

                                    return groups
                                    # print(
                                    # f"[Split Success] Re-grouped by divergence at col {split_col_idx} outside brackets: {list(groups.keys())}")
                                # else:
                                # print(
                                # f"DEBUG: Deep split rejected - no static markers found in column {split_col_idx}")
                    group_logs = groups[first_token_key]

                    def normalize_token(t):

                        return re.sub(r'\.{2,}\d*$', '...', t)


                    log_token_seqs = []
                    for log in group_logs:
                        raw_tokens = log.split()
                        log_token_seqs.append([normalize_token(t) for t in raw_tokens])


                    lengths = set(len(seq) for seq in log_token_seqs)


                    if len(lengths) == 1:
                        log_len = list(lengths)[0]
                        if 3 <= log_len <= 6:
                            split_col = -1

                            for col_idx in range(log_len):
                                col_vals = set(seq[col_idx] for seq in log_token_seqs)
                                if len(col_vals) > 1:
                                    split_col = col_idx
                                    break

                            if split_col != -1:
                                candidate_groups = {}
                                valid_split = True
                                for idx, log in enumerate(group_logs):

                                    orig_val = log.split()[split_col]
                                    if any(char in orig_val for char in "-_<>:+[](){}"):
                                        valid_split = False
                                        break
                                    if '.' in orig_val and '..' not in orig_val:
                                        valid_split = False
                                        break

                                    if len(orig_val) > 3 and re.fullmatch(r'[0-9a-fA-F]+', orig_val):
                                        valid_split = False
                                        break


                                    if any(char.isdigit() for char in orig_val):
                                        valid_split = False
                                        break
                                    norm_val = normalize_token(orig_val)
                                    candidate_groups.setdefault(norm_val, []).append(log)


                                if valid_split and  1 < len(candidate_groups) <= 4:
                                    print(
                                        f"[Split Success] Semantic split found at col {split_col}: {list(candidate_groups.keys())}")
                                    return candidate_groups

        if len(groups) == 1:
            rep_log = log_group[0]

            tokens_count = len(rep_log.split())
            if tokens_count == 2:
                secondary_groups = {}
                for log in log_group:
                    words = log.split()
                    if len(words) < 2: continue

                    sec_token = words[1]

                    norm_token = re.sub(r'\.{2,}\d+$', '...', sec_token)
                    norm_token = re.sub(r'\+0x[0-9a-fA-F]+(?:/0x[0-9a-fA-F]+)?', '+', norm_token)

                    if '=' in sec_token:
                        continue

                    if sec_token.isdigit() or re.fullmatch(r'<.*?\d+.*?>', sec_token):
                        continue


                    if sec_token.count('.') >= 2 or '@' in sec_token:
                        continue


                    if sec_token.lower().startswith('0x') or \
                            (len(sec_token) >= 7 and re.fullmatch(r'[0-9a-fA-F]+', sec_token)):
                        continue


                    secondary_groups.setdefault(norm_token, []).append(log)


                if 1 < len(secondary_groups) <= 10:
                    print(
                        f"[Split Success] Forced split for length-2 logs by 2nd token: {list(secondary_groups.keys())}")
                    return secondary_groups
                elif len(secondary_groups) > 10:

                    return None
                    #print(
                        #f"DEBUG: Too many unique 2nd tokens ({len(secondary_groups)}) in length-2 logs. Likely a variable. Split rejected.")

            if len(groups) < 2:
                return None
        # === Step 2 ===
        is_single_token_scenario = True
        for log in log_group:
            valid_tokens = []
            for m in token_pattern.finditer(log):
                t = m.group(0)
                if len(t) <= 1: continue
                is_path_likely = '/' in log or ':' in log
                if is_path_likely and self._is_inside_path(log, match.start(), match.end()):
                    continue
                if re.match(r'^x[0-9a-fA-F]+$', t) or (len(t) > 3 and re.fullmatch(r'[0-9a-fA-F]+', t)):
                    continue
                valid_tokens.append(t)

            if len(valid_tokens) > 2:
                is_single_token_scenario = False
                break

        if is_single_token_scenario:
            print(f"[Split Detected] Single Token Scenario (e.g., Stack Trace): {list(groups.keys())}")
            return groups

        # --- Step 2.5: Context Signature ---
        group_signatures = {}
        signature_counts = {}
        for token, sub_logs in groups.items():
            if token == "<UNKNOWN>": continue
            rep_log = sub_logs[0]


            sig = self._get_context_signature(rep_log, token)
            group_signatures[token] = sig


            signature_counts[sig] = signature_counts.get(sig, 0) + 1
        # --- Step 2.7: [OPTIMIZED] Pre-calculate Partial Context Matches ---

        partial_match_set = set()

        def is_num_hex(v):
            c = v.strip('.,()[]:;')
            if not c: return False
            return bool(re.match(r'^-?\d+(\.\d+)?$', c) or re.match(r'^0x[0-9a-fA-F]+$', c))


        group_meta = {}
        active_tokens = [t for t in groups.keys() if t != "<UNKNOWN>"]

        for t in active_tokens:
            rep_tokens = groups[t][0].split()
            group_meta[t] = {
                "tokens": rep_tokens,
                "clean": [v.strip('.,()[]:;') for v in rep_tokens],
                "hex_mask": [is_num_hex(v) for v in rep_tokens],
                "len": len(rep_tokens)
            }


        for i, t1 in enumerate(active_tokens):
            meta1 = group_meta[t1]
            for j, t2 in enumerate(active_tokens):
                if i == j: continue
                meta2 = group_meta[t2]


                if meta1["len"] == meta2["len"]:
                    valid_indices = []
                    for idx in range(meta1["len"]):

                        if (meta1["tokens"][idx] == t1 or
                                meta2["tokens"][idx] == t2 or
                                meta1["hex_mask"][idx] or
                                meta2["hex_mask"][idx]):
                            continue
                        valid_indices.append(idx)

                    if valid_indices:
                        match_count = sum(
                            1 for idx in valid_indices if meta1["tokens"][idx] == meta2["tokens"][idx])
                        if match_count / len(valid_indices) > 0.66:
                            partial_match_set.add(t1)
                            break
        # === Step 3: BERT  ===



        is_semantic_slot_at_start = False
        for t, sub_logs in groups.items():
            if t == "<UNKNOWN>": continue
            sig = group_signatures.get(t, "")

            if sub_logs[0].strip().startswith(t) and signature_counts.get(sig, 0) > 1:
                is_semantic_slot_at_start = True
                break
        valid_split_found = False
        for token, sub_logs in groups.items():
            if token == "<UNKNOWN>":
                continue

            representative_log = sub_logs[0]

            if len(token) > 20 or len(token) < 2:
                continue


            base_score = self._calculate_compound_score(representative_log, token)
            is_part_partial_match = token in partial_match_set
            weight = 7.0 if is_part_partial_match else 1.0

            if token.lower().endswith('ing') and len(token) > 4:

                bracket_idx = representative_log.find(']')
                token_idx = representative_log.find(token)
                if bracket_idx != -1 and token_idx > bracket_idx:
                    weight = 1000.0
                    print(f"DEBUG: Detected '-ing' suffix after brackets for token '{token}'. Boosting weight.")

            is_start_of_log = representative_log.strip().startswith(token)
            my_sig = group_signatures.get(token, "")

            has_identical_context = signature_counts.get(my_sig, 0) > 1

            if is_start_of_log:

                token_end_pos = representative_log.find(token) + len(token)
                is_followed_by_immediate_colon = (
                        token_end_pos < len(representative_log) and
                        representative_log[token_end_pos] == ':'
                )


                if has_identical_context and not is_followed_by_immediate_colon:
                    weight = 110.0

                elif is_semantic_slot_at_start and not is_followed_by_immediate_colon:
                    weight = 110.0
                else:

                    weight = 20.0
            elif has_identical_context:
                weight = 40.0
            final_score = base_score * weight
            final_score = min(final_score, 1.0)

            #print(
                #f"DEBUG: Token '{token}' Base: {base_score:.5f} | ContextMatch: {has_identical_context} | Weight: {weight} | Final: {final_score:.4f}",
                #flush=True)
            if final_score >= bert_threshold:
                valid_split_found = True

        if valid_split_found:
            #print(f"[Split Detected] Semantic Branching: {list(groups.keys())}")
            return groups
        else:
            #print("DEBUG: No valid semantic split found.")
            return None
    def _get_optimal_device(self):
        if not torch.cuda.is_available():
            return 'cpu'
        try:
            free_memory = torch.cuda.mem_get_info()[0]
            free_memory_gb = free_memory / (1024 ** 3)
            if free_memory_gb > 4.0:
                return 'cuda'
            else:
                print("[MarkerExtractor] Warning: GPU memory is tight (<4GB). Offloading RoBERTa to CPU.", flush=True)
                return 'cpu'
        except:
            return 'cpu'

    def _compute_jaccard(self, s1, s2):
        set1 = set(s1.split())
        set2 = set(s2.split())
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union else 0.0

    def _smart_sampling(self, log_group, k=30, min_threshold=0.6):

        n = len(log_group)
        if n <= k:
            return log_group
        if n > 500:
            log_group = random.sample(log_group, 500)
            n = 500

        center_idx = n // 2
        center_log = log_group[center_idx]
        center_log_norm = re.sub(r'\.{2,}', '...', center_log)
        candidates = []
        perfect_matches = []

        for log in log_group:
            log_norm = re.sub(r'\.{2,}', '...', log)
            sim = self._compute_jaccard(center_log_norm, log_norm)


            if not log.strip():
                continue


            if sim > 0.99:
                perfect_matches.append(log)

            elif sim >= min_threshold:
                candidates.append(log)


        random.shuffle(candidates)


        selected_logs = candidates[:k]


        if len(selected_logs) < k:
            needed = k - len(selected_logs)

            random.shuffle(perfect_matches)
            selected_logs.extend(perfect_matches[:needed])


        return selected_logs

    def _calculate_compound_score(self, log_text, candidate_word):

        try:

            inputs = self.tokenizer(log_text, return_tensors="pt")
            input_ids = inputs.input_ids.to(self.device)


            cand_ids_a = self.tokenizer.encode(candidate_word, add_special_tokens=False)
            cand_ids_b = self.tokenizer.encode(" " + candidate_word, add_special_tokens=False)

            target_indices = []
            true_ids = []


            def find_sequence_indices(main_ids_tensor, sub_ids_list):
                if not sub_ids_list: return []
                main_len = main_ids_tensor.size(1)
                sub_len = len(sub_ids_list)

                main_ids_list = main_ids_tensor[0].tolist()
                found_indices = []

                for i in range(main_len - sub_len + 1):
                    if main_ids_list[i: i + sub_len] == sub_ids_list:
                        found_indices.extend(range(i, i + sub_len))
                        return found_indices
                return found_indices


            indices = find_sequence_indices(input_ids, cand_ids_a)
            true_ids_found = cand_ids_a


            if not indices:
                indices = find_sequence_indices(input_ids, cand_ids_b)
                true_ids_found = cand_ids_b

            if not indices:
                return 0.0

            target_indices = indices


            current_true_ids = []
            if len(target_indices) == len(true_ids_found):
                current_true_ids = true_ids_found
            else:

                current_true_ids = true_ids_found[:len(target_indices)]


            masked_input_ids = input_ids.clone()
            for idx in target_indices:
                masked_input_ids[0, idx] = self.tokenizer.mask_token_id


            with torch.no_grad():
                outputs = self.model(input_ids=masked_input_ids)
                logits = outputs.logits


            scores = []
            for i, seq_idx in enumerate(target_indices):

                if i >= len(current_true_ids): break
                token_id_gt = current_true_ids[i]

                token_logits = logits[0, seq_idx, :]
                token_probs = torch.softmax(token_logits, dim=-1)


                max_prob = torch.max(token_probs).item()

                prob = token_probs[token_id_gt].item()


                ratio = prob / (max_prob + 1e-12)
                scores.append(ratio)


            if not scores:
                return 0.0


            probs_np = np.array(scores)
            final_score = np.exp(np.mean(np.log(probs_np + 1e-12)))

            return final_score

        except Exception as e:

            import traceback
            traceback.print_exc()
            print(f"DEBUG: Error in V5 score calculation: {e}", flush=True)
            return 0.0


    def _needs_bert_optimization(self, common_tokens, representative_log, sample_logs):



        if '=' in representative_log or ':' in representative_log:
            return True


        #split_pattern = r'[\s=:;,\(\)\[\]\{\}]+'
        split_pattern = r'[\s=:;,\(\)\[\]\{\}\+/]+'
        log_tokens = [t for t in re.split(split_pattern, representative_log) if t]
        total_tokens_count = len(log_tokens)

        if total_tokens_count == 0:
            return False


        if total_tokens_count < 1:
            return False




        common_tokens_count = len(common_tokens)
        static_ratio = common_tokens_count / total_tokens_count


        #if static_ratio > 0.7:
            # print(f"DEBUG: Trigger BERT. Static Ratio: {static_ratio:.2f} ({representative_log[:30]}...)")
            #return True


        return True

    def _is_valid_double_colon_static(self, token, log_content):


        pairs = []
        for m in re.finditer(r'([a-zA-Z0-9_]+)::([a-zA-Z0-9_]+)', log_content):
            pairs.append((m.group(1), m.group(2)))

        if not pairs:
            return False


        dynamic_tokens = set()


        for t1, t2 in pairs:
            has_digit1 = any(c.isdigit() for c in t1)
            has_digit2 = any(c.isdigit() for c in t2)
            if has_digit1 and has_digit2:
                dynamic_tokens.add(t1)
                dynamic_tokens.add(t2)


        for _ in range(len(pairs)):
            changed = False
            for t1, t2 in pairs:
                if t1 in dynamic_tokens and t2 not in dynamic_tokens:
                    dynamic_tokens.add(t2)
                    changed = True
                elif t2 in dynamic_tokens and t1 not in dynamic_tokens:
                    dynamic_tokens.add(t1)
                    changed = True
            if not changed:
                break


        if token in dynamic_tokens:
            return False


        t_idx = log_content.find(token)
        if t_idx != -1:
            pre_text = log_content[:t_idx].rstrip()
            if pre_text:
                last_char = pre_text[-1]
                if last_char == '=':
                    return False
                if last_char == ':' and not pre_text.endswith('::'):
                    if pre_text.count(':') > 2:
                        return False


        for t1, t2 in pairs:
            if token == t1 or token == t2:
                return True

        return False
    def _is_wrapped(self, text, index):
        """
        检测指定位置的 token 是否被单引号或括号包裹
        """
        before = text[:index]

        if before.count("'") % 2 != 0:
            return True

        if before.count("(") > before.count(")"):
            return True
        return False

    def get_static_markers(self, log_group, confidence_threshold=0.1):
        """主入口。"""
        start_t = time.time()
        try:
            return self.get_static_markers_core(log_group, confidence_threshold)
        finally:
            self.total_time += (time.time() - start_t)
    def get_static_markers_core(self, log_group, confidence_threshold=0.1):
        """
        主入口。
        """
        if not log_group:
            return []


        sample_logs = self._smart_sampling(log_group, k=30, min_threshold=0.49)

        #split_pattern = r'[\s=:;,\(\)\[\]\{\}]+'
        split_pattern = r'[\s=:;,\(\)\[\]\{\}\+/"\']+'
        dynamic_blacklist = set()
        if sample_logs:

            tokenized_samples = []
            for log in sample_logs:

                tokens = [t for t in re.split(split_pattern, log) if t]
                tokenized_samples.append(tokens)

            if tokenized_samples:

                ref_len = len(tokenized_samples[0])


                if ref_len > 7:

                    valid_samples = [t for t in tokenized_samples if len(t) == ref_len]


                    if len(valid_samples) > 1:

                        for col_idx in range(ref_len):

                            col_values = set(row[col_idx] for row in valid_samples)


                            if len(col_values) > 2:

                                for val in col_values:
                                    dynamic_blacklist.add(val)

        tokenized_logs = []
        for log in sample_logs:
            tokens = set(t for t in re.split(split_pattern, log) if t)
            tokenized_logs.append(tokens)


        common_tokens = set.intersection(*tokenized_logs)
        final_markers = []

        representative_log = sample_logs[0]

        if not self._needs_bert_optimization(common_tokens, representative_log, sample_logs):
            # print("DEBUG: Skipped BERT (Easy pattern)")
            return []

        # print("DEBUG: Running BERT (Complex/Static pattern detected)")

        has_equal_sign = '=' in representative_log
        has_colon = ':' in representative_log
        Scomposite_identity = {t for match in re.findall(r'\(([a-z]{4}), ([a-z]{4})\)', representative_log) for t in match}
        for token in common_tokens:
            # --- L2 ---
            if not token: continue
            token_idx = representative_log.find(token)
            if token in Scomposite_identity:
                continue
            if '$' in token or '|' in token:
                continue
            if re.fullmatch(r'[A-Z]{2}', token):
                continue
            if any(char.isdigit() for char in token):

                if re.fullmatch(r'[a-z0-9]+', token):
                    if any(c.islower() for c in token):
                        continue

                elif len(token) <= 4:
                    continue

            all_occurrences = [m.start() for m in re.finditer(rf'\b{re.escape(token)}\b', representative_log)]

            if all_occurrences:
                is_contained_in_any_path = False
                for start_pos in all_occurrences:
                    if self._is_inside_path(representative_log, start_pos, start_pos + len(token)):
                        is_contained_in_any_path = True
                        break

                if is_contained_in_any_path:
                    print(f"[Lexical Skip] Token: [{token:<20}] is part of a file path. Result: ❌ DROP")
                    continue

                is_inside_fqdn = False
                for raw_word in representative_log.split():

                    if raw_word.count('.') >= 2 and token in raw_word and raw_word != token:
                        is_inside_fqdn = True
                        break

                if is_inside_fqdn:
                    print(f"[Lexical Skip] Token: [{token:<20}] is inside a FQDN/domain token. Result: ❌ DROP")
                    continue

            if token in dynamic_blacklist:
                continue
            if '-' in token:

                if not re.fullmatch(r'[a-z]-[a-z]', token):
                    continue

            clean_token = token.strip('<>[](){}:,.')


            if not clean_token: continue



            if '.' in token.strip('.'):
                # print(f"[Lexical Skip] Internal dot detected in {token}, dropping.")
                continue


            if not clean_token: continue

            if '.' in clean_token:
                continue


            if clean_token[0].isdigit():
                continue


            if re.fullmatch(r'[0-9a-fA-F]+', clean_token):
                if any(char.isdigit() for char in clean_token):
                    continue
            if len(token) < 2: continue
            if re.match(r'^[\W_]+$', token): continue
            if token.startswith('/') or token.count('/') > 1:
                continue
            if token.lower().startswith('+0x'):
                continue
            if token.lower().startswith('/0x'):
                continue
            if token.isdigit():
                continue
            if token[0].isdigit():
                continue
            if token.count('_') >= 2 and any(c.isdigit() for c in token):
                continue

            is_after_preposition = False
            token_idx = representative_log.find(token)

            PREPOSITION_TRIGGERS = {'from', 'to', 'at', 'by', 'on', 'in', 'for'}


            TEMPORAL_TOKENS = {
                'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun',
                'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
            }
            if token_idx != -1:

                pre_text = representative_log[:token_idx].strip()
                if pre_text:
                    last_word = pre_text.split()[-1].lower()

                    if last_word in PREPOSITION_TRIGGERS :
                        is_after_preposition = True

            is_colon_list_key = False

            if has_colon:

                t_start = representative_log.find(token)
                if t_start != -1:
                    t_end = t_start + len(token)

                    if t_end < len(representative_log):

                        next_part = representative_log[t_end:].lstrip()

                        if next_part.startswith(':'):

                            pre_text = representative_log[:t_start].rstrip()


                            has_valid_comma = False
                            remaining_text = next_part[1:]
                            bracket_level = 0
                            for char in remaining_text:
                                if char == '(':
                                    bracket_level += 1
                                elif char == ')':
                                    bracket_level -= 1
                                elif char == ',' and bracket_level == 0:
                                    has_valid_comma = True
                                    break


                            if (pre_text.endswith('<') or pre_text.endswith(',')) or has_valid_comma:
                                is_colon_list_key = True

            token_idx = -1
            prev_char = ''
            next_char = ''
            context_extracted = False
            is_after_able_word = False
            t_idx = representative_log.find(token)
            if t_idx != -1:

                pre_content_raw = representative_log[:t_idx].rstrip()

                words_before = re.findall(r'[a-zA-Z]+', pre_content_raw)

                if words_before:
                    prev_word = words_before[-1]

                    if prev_word.lower().endswith('able'):

                        pw_idx = pre_content_raw.rfind(prev_word)
                        prefix_of_pw = pre_content_raw[:pw_idx].rstrip()
                        if not prefix_of_pw.endswith('='):

                            if re.fullmatch(r'[a-z]+', clean_token):

                                if not self._is_wrapped(representative_log, t_idx):
                                    is_after_able_word = True
            current_threshold = confidence_threshold
            threshold_reason = "Default"


            is_followed_by_number = False

            if token.islower() and token.isalpha() and len(token) > 2:


                token_idx = representative_log.find(token)
                while token_idx != -1:

                    if token_idx > 0 and representative_log[token_idx - 1].isalpha():
                        token_idx = representative_log.find(token, token_idx + 1)
                        continue


                    token_end = token_idx + len(token)


                    if token_end + 1 < len(representative_log):
                        first_char = representative_log[token_end]
                        second_char = representative_log[token_end + 1]

                        if first_char == ' ' and second_char.isdigit():


                            pre_content_parts = representative_log[:token_idx].strip().split()
                            has_digit_in_prev = False
                            if pre_content_parts:
                                prev_word = pre_content_parts[-1]
                                if any(c.isdigit() for c in prev_word):
                                    has_digit_in_prev = True


                            pre_content_lower = representative_log[:token_idx].rstrip().lower()

                            if not has_digit_in_prev and not pre_content_lower.endswith(" of"):
                                is_followed_by_number = True
                                break


                    token_idx = representative_log.find(token, token_idx + 1)


            if has_equal_sign:
                token_idx = representative_log.find(token)
                if token_idx != -1:

                    for i in range(token_idx - 1, -1, -1):
                        if representative_log[i] != ' ':

                            if representative_log[i] == ':' and i > 0 and representative_log[i - 1] == ':':
                                prev_char = '::'
                            else:
                                prev_char = representative_log[i]
                            break

                    end_idx = token_idx + len(token)
                    for i in range(end_idx, len(representative_log)):
                        if representative_log[i] != ' ':

                            if representative_log[i] == ':' and i + 1 < len(representative_log) and \
                                    representative_log[i + 1] == ':':
                                next_char = '::'
                            else:
                                next_char = representative_log[i]
                            break
                    context_extracted = True
                    if prev_char == '=' or next_char == '=':
                        continue


            is_value_position = False
            is_key_structure = False
            if has_colon:
                if not context_extracted:
                    token_idx = representative_log.find(token)
                    if token_idx != -1:

                        for i in range(token_idx - 1, -1, -1):
                            if representative_log[i] != ' ':
                                if representative_log[i] == ':' and i > 0 and representative_log[i - 1] == ':':
                                    prev_char = '::'
                                else:
                                    prev_char = representative_log[i]
                                break
                        end_idx = token_idx + len(token)
                        for i in range(end_idx, len(representative_log)):
                            if representative_log[i] != ' ':
                                if representative_log[i] == ':' and i + 1 < len(representative_log) and \
                                        representative_log[i + 1] == ':':
                                    next_char = '::'
                                else:
                                    next_char = representative_log[i]
                                break
                        context_extracted = True

                if context_extracted:
                    try:

                        if prev_char == ':':
                            is_value_position = True
                        if next_char == ':':
                            is_value_position = False
                    except:
                        pass

            def is_title_case_label(t):

                return re.match(r'^[A-Z][a-z]+:$', t) is not None

            def is_complex_token(t):

                return ('\\' in t and
                        re.search(r'[A-Z]', t) and
                        re.search(r'[a-z]', t) and
                        re.search(r'[0-9]', t))
            # --- L3: RoBERTa  ---
            token_for_scoring = clean_token.rstrip('.')
            score = self._calculate_compound_score(representative_log, token_for_scoring)
            #score = self._calculate_compound_score(representative_log, clean_token)
            is_temporal_entity = token.lower().strip(',.') in TEMPORAL_TOKENS


            current_threshold = confidence_threshold
            threshold_reason = "Default"
            lower_clean = clean_token.lower()
            if self._is_valid_static_ed_word(token, representative_log):
                current_threshold = confidence_threshold * 0.001
                threshold_reason = "Suffix (ed) "
            #if self._is_valid_static_id_word(token, representative_log):
                #current_threshold = confidence_threshold * 0.1
                #threshold_reason = "Suffix (id) - High Priority"
            if self._is_valid_static_ty_word(token, representative_log):
                current_threshold = confidence_threshold * 0.0001
                threshold_reason = "Suffix (ty) "
            if lower_clean.endswith(('ful', 'lly')):
                current_threshold = 0.0
                threshold_reason = "Suffix (ful/lly) "
            elif lower_clean.endswith('ly'):

                current_threshold = confidence_threshold * 0.01
                threshold_reason = "Suffix (ly) - High Priority"
            if is_after_able_word:
                current_threshold = confidence_threshold * 0.0005
                threshold_reason = "After -able word (Static Pair)"
            if is_temporal_entity:

                current_threshold = 1.1
                threshold_reason = "Temporal Entity (Forced Var)"
            elif is_after_preposition:
                current_threshold = confidence_threshold * 2.6  # 0.26
                threshold_reason = "After Preposition"
            if re.fullmatch(r'[a-z]-[a-z]', token):
                current_threshold = confidence_threshold * 0.01
            match_next = re.search(rf'\b{re.escape(token)}\b\s+([^\s,;:]+)', representative_log)
            next_token = match_next.group(1) if match_next else ""


            if is_title_case_label(token):
                if is_complex_token(next_token):
                    current_threshold = 1.1
                    threshold_reason = "Label Before Complex Token (Forced Var)"
            if self._is_valid_double_colon_static(token, representative_log):
                current_threshold = confidence_threshold * 0.0001
                threshold_reason = "Double Colon (::) "
            elif is_value_position:
                current_threshold = confidence_threshold * 6.0
                threshold_reason = "Value Pos (:)"
            elif is_key_structure:
                current_threshold = 0.001
                threshold_reason = "Key Structure"
            elif is_followed_by_number:
                if threshold_reason == "Default":
                    current_threshold = 0.001
                    threshold_reason = "Followed by Number"
            elif is_colon_list_key:
                current_threshold = 0.001
                threshold_reason = "Colon List Key"

            if threshold_reason == "Default" and ('.' in token.rstrip('.') or '_' in token) and not is_value_position:
                if not is_followed_by_number:
                    current_threshold = confidence_threshold * 0.65
            status = "✅ KEEP" if score > current_threshold else "❌ DROP"
            print(
                f"[BERT Score] Token: [{token_for_scoring:<20}] Score: {score:.6f} | Threshold: {current_threshold:.6f} ({threshold_reason}) | Result: {status}",
                flush=True)
            if score > current_threshold:
                final_markers.append(clean_token)

        return final_markers