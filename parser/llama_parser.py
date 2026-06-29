import os
import sys
import argparse
import torch
import difflib
import csv
from vllm import SamplingParams
import json
import re
import ast
import pandas as pd
import time
import random
import textdistance
import transformers
import regex_manager
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from random import shuffle
from datetime import datetime


def replace_bracketed_uppercase(text):
    pattern = r'<[A-Z_]+>'
    replaced_text = re.sub(pattern, '<*>', text)
    return replaced_text.strip()


def get_logs_from_group(group_list):
    logs_from_group = []
    for ele in group_list:
        logs_from_group.append(ele["Content"])
    return logs_from_group


def verify_one_regex(log, regex):
    log = log.replace(",", "").replace("$", "")
    regex = regex.replace(",", "").replace("$", "")
    try:
        if re.fullmatch(regex, log):
            return True
        else:
            return False
    except re.error as e:
        return False


def check_and_truncate_regex(pattern):
    parts = re.split(r'(\(\.\*\?\))', pattern)
    wildcards_count = parts.count('(.*?)')

    if wildcards_count > 30:
        index_30th = [i for i, part in enumerate(parts) if part == '(.*?)'][29]
        truncated_parts = parts[:index_30th + 1]
        truncated_pattern = ''.join(truncated_parts)
        return truncated_pattern
    else:
        return pattern


class LogParser:
    def __init__(
            self,
            pipeline,
            regex_manager1,
            model="Meta-Llama-3-8B-Instruct",
            regex_sample=5,
            similarity="jaccard",
            do_self_reflection="True"
    ):
        self.total_time = 0.0
        self.new_event = 0
        self.model = model
        self.regex_sample = regex_sample
        self.pipeline = pipeline
        self.regex_manager1 = regex_manager1
        self.similarity = similarity
        self.do_self_reflection = do_self_reflection
        self.unmatched_logs = []
        self.tokenizer = self.pipeline.get_tokenizer()
        print("parser is ready.", flush=True)

    def _abstract_dynamic_structural_tokens(self, template, log_list):
        if not log_list or len(log_list) < 2:
            return template

        first_log = log_list[0]["Content"] if isinstance(log_list[0], dict) else log_list[0]

        unit_pattern_str = r'\d+(?:\.\d+)?\s+(?:B|KB|MB|GB|TB|PB|G|K|M)\b'
        if not re.search(unit_pattern_str, first_log):
            return template

        t_pattern = re.compile(r'(?:\<\*\>|\(\.\*\?\)|VAR_HOLDER|\d+(?:\.\d+)?)(?:\.\*)?\s+(B|KB|MB|GB|TB|PB|G|K|M)\b')
        t_matches = list(t_pattern.finditer(template))
        if not t_matches:
            return template

        total_logs = len(log_list)
        max_check = 1000
        step = max(1, total_logs // max_check)
        check_indices = [i * step for i in range(min(max_check, total_logs))]
        if (total_logs - 1) not in check_indices:
            check_indices.append(total_logs - 1)

        clean_template = template.replace(r'\(', '(').replace(r'\)', ')').replace(r'\.', '.').replace(r'\-', '-')
        refined_template = clean_template
        to_replace = []

        for t_match in t_matches:
            unit_in_template = t_match.group(1)
            start_idx, end_idx = t_match.start(1), t_match.end(1)

            prefix_text = clean_template[:t_match.start()].strip()
            anchors = re.findall(r'\b[a-zA-Z]{3,}\b', prefix_text)
            anchor = anchors[-1] if anchors else None

            is_dynamic = False
            for idx in check_indices:
                content = log_list[idx]["Content"] if isinstance(log_list[idx], dict) else log_list[idx]
                content = content.replace(",", "").replace("$", "")

                current_log_unit = None
                if anchor:
                    search_pat = rf'{re.escape(anchor)}' + r'[^a-zA-Z]*?\d+(?:\.\d+)?\s+([A-Z]{1,2})\b'
                    log_match = re.search(search_pat, content)
                    if log_match:
                        current_log_unit = log_match.group(1)
                else:
                    all_units = re.findall(r'\d+(?:\.\d+)?\s+([A-Z]{1,2})\b', content)
                    if all_units:
                        current_log_unit = all_units[0]

                if current_log_unit and current_log_unit != unit_in_template:
                    is_dynamic = True
                    break

            if is_dynamic:
                to_replace.append((start_idx, end_idx))

        for start, end in reversed(to_replace):
            refined_template = refined_template[:start] + "<*>" + refined_template[end:]

        return refined_template

    def cosine_similarity_distance(self, x, y):
        vectorizer = CountVectorizer()
        x_vector = vectorizer.fit_transform([x])
        y_vector = vectorizer.transform([y])
        similarity = cosine_similarity(x_vector, y_vector)[0][0]
        return 1 - similarity

    def jaccard_distance(self, x, y):
        return textdistance.jaccard.normalized_distance(x.split(), y.split())

    def min_distance(self, c_set, t_set):
        D = []
        for c_inst in c_set:
            min_candidate_distance = 1e10
            for t_inst in t_set:
                if self.similarity == "cosine":
                    min_candidate_distance = min(min_candidate_distance,
                                                 self.cosine_similarity_distance(c_inst, t_inst))
                elif self.similarity == "jaccard":
                    min_candidate_distance = min(min_candidate_distance, self.jaccard_distance(c_inst, t_inst))
                else:
                    raise ValueError("Invalid similarity metric.")
            D.append(min_candidate_distance)
        return D

    def adaptive_random_sampling(self, logs, k, max_logs=200, similarity_flag=False, dic=False):
        if dic:
            logs = get_logs_from_group(logs)

        if max_logs is not None and len(logs) > max_logs:
            logs = random.sample(logs, max_logs)

        if len(logs) < k:
            k = len(logs)
        sample_list = []
        T = []
        if self.similarity == "random":
            sample_list = random.sample(logs, k)
        else:
            for r in range(k):
                if len(sample_list) == 0:
                    i = max(range(len(logs)), key=lambda x: len(logs[x].split()))
                    T.append(logs[i])
                    sample_list.append(logs[i])
                    del logs[i]
                else:
                    candidate_distance = self.min_distance(logs, T)
                    if similarity_flag:
                        best_candidate = min(range(len(candidate_distance)), key=lambda x: candidate_distance[x])
                    else:
                        best_candidate = max(range(len(candidate_distance)), key=lambda x: candidate_distance[x])

                    T.append(logs[best_candidate])
                    sample_list.append(logs[best_candidate])
                    logs.remove(logs[best_candidate])

        return sample_list

    def _generate_correction_context(self, original_template, failed_log):
        """生成包含错误原因的校正上下文"""
        pseudo_template = re.sub(r'\(\.\*\?\)', '<*>', original_template)
        clean_pseudo_template = pseudo_template
        chars_to_unescape = [r'\.', r'\-', r'\:', r'\[', r'\]', r'\(', r'\)', r'\$']
        for char in chars_to_unescape:
            clean_pseudo_template = clean_pseudo_template.replace(char, char.replace('\\', ''))

        def _find_hallucinated_content(log, template):
            temp_text = template.replace('<*>', ' ')
            temp_text = temp_text.replace(r'\.', '.').replace(r'\-', '-')
            potential_words = re.findall(r'[a-zA-Z0-9_\-\.]{3,}', temp_text)
            hallucinations = []
            log_lower = log.lower()
            for word in potential_words:
                if word.isdigit():
                    continue
                if word in ["..."]:
                    continue
                if word.lower() not in log_lower:
                    clean_word = word.strip(".,=:")
                    if len(clean_word) > 2 and clean_word.lower() not in log_lower:
                        hallucinations.append(word)
            return list(set(hallucinations))

        hallucinated_words = _find_hallucinated_content(failed_log, pseudo_template)

        def _find_untreated_paths(original_log, template):
            path_pattern = r'((?:https?://|www\.|/)[a-zA-Z0-9._/+-]+)'
            paths = re.findall(path_pattern, original_log)
            untreated = []
            clean_template = template.replace(r'\-', '-').replace(r'\.', '.')
            for path in paths:
                if len(path) < 3: continue
                if path in clean_template:
                    continue
                path_parts = re.split(r'/', path)
                for part in path_parts:
                    if len(part) < 2: continue
                    if part in clean_template:
                        untreated.append(path)
                        break
                    if '-' in part:
                        suffix = part.split('-')[-1]
                        if len(suffix) > 0 and (f"-{suffix}" in clean_template) and (f"-{suffix}" in part):
                            untreated.append(path)
                            break
            return list(set(untreated))

        correction_msg = f"""CRITICAL ERROR: The generated template is INCORRECT.

    REQUIRED FIXES:
    """
        if hallucinated_words:
            correction_msg += f"[HALLUCINATION DETECTED] You added content that DOES NOT EXIST in the original log.\n"
            correction_msg += f"   - The original log is: {failed_log}\n"
            correction_msg += f"   - You incorrectly added: {', '.join(hallucinated_words)}\n"
            correction_msg += f"   - ACTION: REMOVE these non-existent parts immediately. The template must ONLY match the log content.\n"

        path_trigger_pattern = r'(?:^|[\s:=\'"\[\(])(?:/|https?://|www\.)[a-zA-Z0-9._/-]+'
        if re.search(path_trigger_pattern, failed_log):
            untreated_paths = _find_untreated_paths(failed_log, pseudo_template)
            if untreated_paths:
                correction_msg += "[Format Error] The following text is a SINGLE PATH string. Do NOT split it. Abstract the WHOLE path as one variable:\n"
                for p in untreated_paths:
                    correction_msg += f"   - Path: {p}\n"
                    correction_msg += f"for example  WRONG: .../-2/...\n"
                    correction_msg += f"             RIGHT: <*>\n"

        if '=' in failed_log or ':' in failed_log:
            hardcoded_params = self.detect_hardcoded_parameters(original_template)
            if hardcoded_params:
                correction_msg += "[Abstraction Error] You interpreted parameters as constants. Any text after '=' or ':' is usually a variable.\n"
                for param in hardcoded_params:
                    if '=' in param:
                        key, val = param.split('=', 1)
                        correction_msg += f"   - Found: {param}\n"
                        correction_msg += f"     ACTION: Change it to {key}=<*>\n"
                    elif ':' in param:
                        key, val = param.split(':', 1)
                        correction_msg += f"   - Found: {param}\n"
                        correction_msg += f"     ACTION: Change it to {key}: <*>\n"

        correction_msg += f"""
    FINAL INSTRUCTION:
    - Keep the structure exactly as the Original Log.
    - Abstract actual variables (such as IDs, paths, or numbers)
    """
        return correction_msg

    def detect_coverage_issues(self, log, regex):
        log_clean = log.replace(",", "")
        regex_clean = regex.replace(",", "")
        strict_regex = regex_clean if regex_clean.endswith('$') else regex_clean + '$'
        try:
            if re.search(strict_regex, log_clean):
                return None
        except:
            pass

        loose_regex = regex_clean.rstrip('$')
        try:
            match = re.search(loose_regex, log_clean)
            if not match:
                return None
            issues = []
            if match.start() > 0:
                missed_head = log_clean[:match.start()].strip()
                if len(missed_head) > 1:
                    issues.append(f"beginning text: '{missed_head}'")
            if match.end() < len(log_clean):
                missed_tail = log_clean[match.end():].strip()
                if len(missed_tail) > 1:
                    issues.append(f"ending text: '{missed_tail}'")
            return issues
        except:
            return None

    def _get_structural_representation(self, text):
        structural = re.sub(r'<\*>', '[V]', text)
        structural = re.sub(r'([,;:])', r'[\1]', structural)
        if len(structural) > 80:
            structural = structural[:80] + "..."
        return structural

    def _inject_correction_into_prompt(self, base_prompt, correction_context, is_chatglm=False, is_mistral=False,
                                       is_qwen=False):
        if is_chatglm:
            base_prompt.insert(0, {
                "role": "system",
                "content": correction_context
            })
            return base_prompt
        elif is_mistral:
            base_prompt.insert(-1, {
                "role": "user",
                "content": correction_context
            })
            return base_prompt
        elif is_qwen:
            base_prompt[0]["content"] = correction_context + "\n\n" + base_prompt[0]["content"]
            return base_prompt
        else:
            base_prompt[0]["content"] = correction_context + "\n" + base_prompt[0]["content"]
            return base_prompt

    def generate_prompt_with_log_list(self, log_list, dic=False, correction_context=None):
        trimmed_list_log = self.adaptive_random_sampling(log_list, self.regex_sample, dic=dic)
        proactive_tips = ""
        if correction_context is None:
            proactive_tips = self._get_proactive_instructions(trimmed_list_log)

        system_prompt = f"""You will be provided with a list of logs.You must identify and abstract all the dynamic variables in logs with ‘<*>‘ and output ONE static log template that matches all the logs.
    {proactive_tips}
    "Conservative Abstraction Rule: Always abstract parts that function as semantically unimportant variables (like numbers, IDs, IPs, network interfaces, and hardware slots). If a word (like 'failed', 'error', service names) appears in ALL provided logs, you may should keep it as static text in the template. Do NOT over-generalize."
    Print the input logs’ template delimited by backticks"""

        message = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": 'Log list: ["service_check(sshd:auth): user unknown; status ok.", "service_check(sshd:auth): user unknown; status ok."]',
            },
            {
                "role": "assistant",
                "content": "`service_check(sshd:auth): user unknown; status ok.`",
            },
            {
                "role": "user",
                "content": 'Log list: ["1 more do_m_failed_file_1_page+123(0x404)", "1 more do_m_failed_file_1_page+123(0x404)", "1 more do_m_failed_file_1_page+123(0x404)"]',
            },
            {
                "role": "assistant",
                "content": "`<*> more do_m_failed_file_1_page+<*>(<*>)`",
            },
            {
                "role": "user",
                "content": 'Log list: ["ipv4 Connection failed error code : #1 [mail.net-com.cn] 1.1-2.3 -456 (abcd-75-66.pacbell.baidu.net) ...", "ipv4 Connection failed error code : #1 [mail.net-com.cn] 1.1-2.3 -123 (abcd-75-66.pacbell.baidu.net) ..."]',
            },
            {
                "role": "assistant",
                "content": "`ipv4 Connection failed error code : <*> [<*>] <*> <*> (<*>) ...`",
            },
            {
                "role": "user",
                "content": 'Log list: ["v2.0.0 Received connect from 10.0.0.1: 11: Hello .", "v2.0.0 Received connect from 192.168.1.5: 999: Hello ."]',
            },
            {
                "role": "assistant",
                "content": "`<*> Received connect from <*>: <*>: Hello .`",
            },
            {
                "role": "user",
                "content": 'Log list: ["TRACE: 987.654: vmnic2: sync_protocol_metrics: Initial Seq: 1000, Val: 2000, MTU: 1500", "TRACE: 987.654: vmnic2: sync_protocol_metrics: Initial Seq: 1000, Val: 2000, MTU: 1500"]',
            },
            {
                "role": "assistant",
                "content": "`TRACE: <*>: <*>: sync_protocol_metrics: Initial Seq: <*>, Val: <*>, MTU: <*>`",
            },
            {"role": "user", "content": f"Log list: {trimmed_list_log}"},
        ]

        if correction_context:
            message = self._inject_correction_into_prompt(message, correction_context)

        full_prompt = self.tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        return full_prompt, trimmed_list_log

    def generate_prompt_with_log_list_qwen(self, log_list, dic=False, correction_context=None):
        trimmed_list_log = self.adaptive_random_sampling(log_list, self.regex_sample, dic=dic)
        proactive_tips = ""
        if correction_context is None:
            proactive_tips = self._get_proactive_instructions(trimmed_list_log)

        system_prompt = f"""You are an expert system for log parsing. You will be provided with a list of logs. You must identify and abstract all the dynamic variables in logs with '<*>' and output ONE static log template that matches all the logs.
{proactive_tips}
Conservative Abstraction Rule: Always abstract parts that function as semantically unimportant variables (like numbers, IDs, IPs, network interfaces, and hardware slots). If a word (like 'failed', 'error', service names) appears in ALL provided logs, you should keep it as static text in the template. Do NOT over-generalize.
Print the input logs' template delimited by backticks."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",
             "content": 'Log list: ["v2.0.0 ipv4 error code : [mail.net-com.cn] 1.1-2.3 (abcd-75-66.pacbell.baidu.net) Received connect from 10.0.0.1: 11: Hello .", "v2.0.0 ipv4 error code : [mail.net-com.cn] 1.1-2.3 (abcd-75-66.pacbell.baidu.net) Received connect from 192.168.1.5: 999: Hello ."]'},
            {"role": "assistant",
             "content": "`<*> ipv4 error code : [<*>] <*> (<*>) Received connect from <*>: <*>: Hello .`"},
            {"role": "user",
             "content": 'Log list: ["TRACE: 987.654: vmnic2: sync_protocol_metrics: Initial Seq: 1000, Val: 2000, MTU: 1500", "TRACE: 987.654: vmnic2: sync_protocol_metrics: Initial Seq: 1000, Val: 2000, MTU: 1500"]'},
            {"role": "assistant",
             "content": "`TRACE: <*>: <*>: sync_protocol_metrics: Initial Seq: <*>, Val: <*>, MTU: <*>`"},
            {"role": "user",
             "content": 'Log list: ["- Sandbox: com.apple.SafariHelper(4921) deny(1) file-read-data /Library/Preferences/com.apple.security.plist opts=(null) confidence=1", "- Sandbox: com.apple.SafariHelper(5032) deny(2) file-read-data /Users/admin/Library/Caches/com.apple.Safari/Cache.db opts=(0x12) confidence=0"]'},
            {"role": "assistant",
             "content": "`- Sandbox: com.apple.SafariHelper(<*>) deny(<*>) file-read-data <*> opts=(<*>) confidence=<*>`"},
            {"role": "user", "content": f"Log list: {trimmed_list_log}"},
        ]

        if correction_context:
            messages = self._inject_correction_into_prompt(messages, correction_context, is_qwen=True)

        full_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return full_prompt, trimmed_list_log

    def generate_prompt_with_log_list_chatglm(self, log_list, dic=False, correction_context=None):
        trimmed_list_log = self.adaptive_random_sampling(log_list, self.regex_sample, dic=dic)
        messages = [
            {
                "role": "system",
                "content": """You will be provided with a list of logs. You must identify and abstract ALL the dynamic variables in logs with ‘<VARIABLE>‘ and output ONLY ONE static log template that matches all the logs in the log list. Print the input logs’ template delimited by backticks.""",
            },
            {
                "role": "user",
                "content": 'Log list: ["try to connected to host: 172.16.254.1, finished.", "try to connected to host: 173.16.254.2, finished."]',
            },
            {
                "role": "assistant",
                "content": "Log Template: `try to connected to host: <VARIABLE>, finished.`",
            },
            {"role": "user", "content": f'Log list: {trimmed_list_log}'},
        ]

        if correction_context:
            messages = self._inject_correction_into_prompt(messages, correction_context, is_chatglm=True)

        return messages, trimmed_list_log

    def generate_prompt_with_log_list_mistral(self, log_list, dic=False, correction_context=None):
        trimmed_list_log = self.adaptive_random_sampling(log_list, self.regex_sample, dic=dic)
        messages = [
            {
                "role": "user",
                "content": """You will be provided with a list of logs. You must identify and abstract all the dynamic variables in logs with ‘<*>‘ and output ONE static log template that matches all the logs. Print the input logs’ template delimited by backticks""",
            },
            {
                "role": "assistant",
                "content": "OK!",
            },
            {
                "role": "user",
                "content": 'Log list: ["try to connected to host: 172.16.254.1, finished.", "try to connected to host: 173.16.254.2, finished."]',
            },
            {
                "role": "assistant",
                "content": "`try to connected to host: <*>, finished.`",
            },
            {
                "role": "user",
                "content": 'Log list: ["Received connect from 10.0.0.1: 11: Hello.", "Received connect from 192.168.1.5: 999: Hello."]',
            },
            {
                "role": "assistant",
                "content": "`Received connect from <*>: <*>: Hello.`",
            },
            {"role": "user", "content": f"Log list: {trimmed_list_log}"},
        ]

        if correction_context:
            messages = self._inject_correction_into_prompt(messages, correction_context, is_mistral=True)

        full_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return full_prompt, trimmed_list_log

    def check_pre_logs(self, log_list, dic=False):
        if dic:
            log_list = get_logs_from_group(log_list)
        log_unique_list = list(set(log_list))
        first_element = log_unique_list[0]
        if len(log_unique_list) != 1:
            return False
        elif (
                (" is " in first_element)
                or ("=" in first_element)
                or (" to " in first_element)
                or ("_" in first_element)
                or ("-" in first_element)
                or (":" in first_element)
                or ("." in first_element)
                or any(char.isdigit() for char in first_element)
        ):
            return False
        return True

    def check_long_logs(self, log_list, dic=False):
        if dic:
            log_list = get_logs_from_group(log_list)
        if len(log_list[0].split()) > 100 and verify_one_regex(log_list[0],
                                                               "Warning: we failed to resolve data source name (.*?)$"):
            return True
        return False

    def template_to_regex(self, template):
        template = template.strip()
        if "chatglm" in self.model:
            template = replace_bracketed_uppercase(template.replace("Log Template: ", "").strip())
        while template.startswith("```") and template.endswith("```"):
            template = template[4:-4]
        while template.startswith("`"):
            template = template[1:]
        while template.endswith("`"):
            template = template[:-1]
        while template.endswith("<*"):
            template = template + ">"
        while template.endswith("<"):
            template = template + "*>"
        while template.endswith("\\"):
            template = template[:-1]
        num_var_pattern = r'(^|[\s:;=\(\)\[\],])[-+]?\d+(?:\.\d+)?(?:%|[a-zA-Z]{1,5})?(?=[\s:;=\(\)\[\],]|$)'
        while "<*>##<*>" in template:
            template = template.replace("<*>##<*>", "<*>")
        template = re.sub(r'<\*>\s*\+\s*<\*>', '<*>', template)
        template = re.sub(num_var_pattern, r'\1<*>', template)
        template = re.sub(r'\<\*\d+\*\>', "<*>", template)
        template = re.sub(r'\<\*\d+\*', "<*>", template)
        template = re.sub(r'\<\*\d+', "<*>", template)
        template = re.sub(r'\<\*\d+\*\>', "<*>", template)
        template = template.replace('*<>', "<*>").replace('*<*>', "<*>").replace('<*>*', "<*>").replace('<>*',
                                                                                                        "<*>").replace(
            '<*|*>', "<*>").replace('<*1*>', "<*>").replace('<>', "<*>")
        template = template.replace("<*>.*", "<*>")
        template = re.sub(r'\b(\w+)\s*=\s*\[\]', r'\1=<*>', template)
        template = re.sub(r'\(\s*\*\s*\)', '(<*>)', template)
        template = re.sub(r'<(\*)(?!>)', '<*>', template)
        template = re.sub(r'(?<!<)(\*)>', '<*>', template)

        while True:
            old_template = template
            template = re.sub(r'<\*>_[a-zA-Z0-9\-]+_<\*>', '<*>', template)
            template = re.sub(r'<\*>\-[a-zA-Z0-9\-]+\-<\*>', '<*>', template)
            template = re.sub(r'\b[a-zA-Z0-9]+_<\*>', '<*>', template)
            template = re.sub(r'\b[a-zA-Z0-9]+\-<\*>', '<*>', template)
            template = re.sub(r'<\*>_[a-zA-Z0-9]+\b', '<*>', template)
            template = re.sub(r'<\*>\-[a-zA-Z0-9]+\b', '<*>', template)
            template = re.sub(r'<\*>_<\*>', '<*>', template)
            template = re.sub(r'<\*>\s*<\*>', '<*>', template)
            template = re.sub(r'<\*>\-<\*>', '<*>', template)
            template = re.sub(r'\(\s*<\*>\s*\)\s*\(\s*<\*>\s*\)', '(<*>)', template)
            template = re.sub(r'<\*>:[a-zA-Z0-9\-]+:<\*>', '<*>', template)
            if template == old_template:
                break

        template = re.sub(r"'<\*>'?", "<*>", template)
        escaped = re.escape(template)
        regex_pattern = re.sub(r'<\\\*>', r'(.*?)', escaped)
        regex_pattern = re.sub(r'(\(\.\*\?\))+', r'(.*?)', regex_pattern)
        regex_pattern = regex_pattern.replace("\ ", " ")
        regex_pattern = re.sub(r'(\(\.\*\?\)){2,}', r'(.*?)', regex_pattern)
        regex_pattern = re.sub('(\(\.\*\?\) ){10,}', '(.*?) (.*?) (.*?) (.*?) (.*?) (.*?) (.*?) (.*?) (.*?) (.*?)',
                               regex_pattern, 0)
        regex_pattern = check_and_truncate_regex(regex_pattern)
        return regex_pattern

    def generalize_regex(self, target_string, regex_pattern):
        try:
            option_patterns = re.findall(r'\(\?\:(.*?)\)', regex_pattern)
            for option_pattern in option_patterns:
                options = option_pattern.split('|')
                for option in options:
                    modified_pattern = regex_pattern.replace(f"(?:{option_pattern})", option)
                    if re.match(modified_pattern, target_string):
                        return modified_pattern
        except:
            return regex_pattern
        return regex_pattern

    def correct_single_template(self, template, log=None):
        template = re.sub(r'0[xX][0-9a-fA-F]+', '(.*?)', template)
        path_delimiters = {
            r'\s', r'\,', r'\!', r'\;', r'\:',
            r'\=', r'\|', r'\"', r'\'',
            r'\[', r'\]', r'\(', r'\)', r'\{', r'\}', r'/'
        }
        token_delimiters = path_delimiters.union({
            r'\.', r'\-', r'\+', r'\@', r'\#', r'\$', r'\%', r'\&',
        })
        template = template.replace("proxy\.((?:[^.]+|\.)*(?:\.-?\d+)+):[0-9]+", "(.*?)")
        template = template.replace("\proxy\.([^.]+):(?:-?\d+|443)", "(.*?)")
        template = template.replace("(?:.*?:-?\d+)?", "(.*?)")
        template = template.replace("(.*?|.*)", "(.*?)")
        template = template.replace("(?:\\n|$)", "$")
        template = template.replace("(\\b)", "").replace("\\b", "").replace("(\\n)", "").replace("\\n", "").replace(
            "(?i)",
            "").replace(
            "?i", "").replace("(\\r)", "").replace("\\r", "")

        template = re.sub(r'\s+[`\'"]\(\.\*\?\)[`\'"]\s+', r' (.*?) ', template)
        template = re.sub(r'\b[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F]+\b', '(.*?)', template)
        while re.search(r'\b[a-zA-Z](\(\.\*\?\))', template):
            template = re.sub(r'\b[a-zA-Z](\(\.\*\?\))', r'\1', template)
        while re.search(r'([a-zA-Z0-9_])(\(\.\*\?\))', template):
            match = re.search(r'([a-zA-Z0-9_])(\(\.\*\?\))', template)
            template = re.sub(r'([a-zA-Z0-9_])(\(\.\*\?\))', r'(.*?)', template)
        template = re.sub(r'\b([0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}\b', '(.*?)', template)
        template = template.strip()
        template = re.sub(r'\s+', ' ', template)

        while re.search(r'(\(\.\*\?\))(?:\\\.[a-zA-Z0-9_-]+){1,}', template):
            template = re.sub(r'(\(\.\*\?\))(?:\\\.[a-zA-Z0-9_-]+){1,}', r'(.*?)', template)

        if log:
            def patch_empty_key_value(match):
                full_match = match.group(0)
                key_eq_part = match.group(1)

                key_name_match = re.search(r'([a-zA-Z0-9_.-]+)', key_eq_part)
                if not key_name_match:
                    return full_match
                clean_key = key_name_match.group(1)

                empty_check_pattern = rf'\b{re.escape(clean_key)}\s*=\s*(?=[a-zA-Z0-9_.-]+=|[ \t,;\]\)]|$)'
                log_match = re.search(empty_check_pattern, log)

                if log_match:
                    actual_key_eq = log_match.group(0).rstrip()
                    return actual_key_eq + "(.*?)"

                return full_match

            template = re.sub(r'(\b[a-zA-Z0-9_.-]+\s*=\s+)\(\.\*\?\)', patch_empty_key_value, template)

        template = re.sub(r'0[xX]\s*(\(\.\*\?\))', r'(.*?)', template)
        template = re.sub(r'0[xX]\s*<.*?>', r'(.*?)', template)

        while re.search(r'\b([A-Z]{4,5})/(\(\.\*\?\))', template):
            template = re.sub(r'\b([A-Z]{2,5})/(\(\.\*\?\))', r'(.*?)', template)

        while re.search(r'[a-zA-Z0-9_$.\\\\]+@\(\.\*\?\)', template):
            template = re.sub(r'[a-zA-Z0-9_$.\\\\]+@\(\.\*\?\)', r'(.*?)', template)

        protocol_uri_pattern = r'[a-zA-Z0-9]+\\?:/{1,3}(?:(?:\(\.\*\?\))|[^\s+),])*?(?:(?:\(\.\*\?\))|[^\s+),.])'
        while re.search(protocol_uri_pattern, template):
            template = re.sub(protocol_uri_pattern, r'(.*?)', template)

        rel_path = r'(?:^|[\s:=])([a-zA-Z0-9._-]*/(?:(?:\(\.\*\?\))|[^\s+),])*?(?:(?:\(\.\*\?\))|[^\s+),.]))'
        for match in re.finditer(rel_path, template):
            path_candidate = match.group(1)
            if path_candidate.count('/') >= 2:
                template = template.replace(path_candidate, '(.*?)')

        if log:
            def protect_token_dot_var(match):
                token = match.group(1)
                full_pattern = match.group(0)
                log_check_pattern = rf'\b{re.escape(token)}\.((?:\d+(?:\.\d+)?)|(?:<\d+>))'

                if re.search(log_check_pattern, log):
                    return full_pattern
                else:
                    return "(.*?)"

            template = re.sub(r'(?<!\.)\b([a-zA-Z0-9]+)\\\.(\(\.\*\?\))', protect_token_dot_var, template)
            float_mis_pattern = r'\(\.\*\?\)\\\.\s*\(\.\*\?\)'

            def float_merger_callback(match):
                full_match_text = match.group(0)
                if re.search(r'\d+\.\d+', log):
                    return "(.*?)"
                return full_match_text

            template = re.sub(float_mis_pattern, float_merger_callback, template)

        while re.search(r'(?:[a-zA-Z0-9_-]+\\\.){2,}(\(\.\*\?\))', template):
            template = re.sub(r'(?:[a-zA-Z0-9_-]+\\\.){2,}(\(\.\*\?\))', r'\1', template)

        double_colon_chain_pattern = r'(\(\.\*\?\))::[a-zA-Z0-9_]+::[a-zA-Z0-9_]+'
        while re.search(double_colon_chain_pattern, template):
            template = re.sub(double_colon_chain_pattern, r'(.*?)', template)

        while re.search(r'([a-zA-Z0-9_\-]+)(?<![=:])(\(\.\*\?\))', template):
            template = re.sub(r'([a-zA-Z0-9_\-]+)(?<![=:])(\(\.\*\?\))', r'(.*?)', template)

        while re.search(r'(\(\.\*\?\))([a-zA-Z0-9_\-\.]+)(?!\\?[=:])', template):
            template = re.sub(r'(\(\.\*\?\))([a-zA-Z0-9_\-\.]+)(?!\\?[=:])', r'(.*?)', template)

        win_path_pattern = r'\(\.\*\?\):\s*(?:\\+\s*(?:\(\.\*\?\)|[a-zA-Z0-9_]+))+'
        while re.search(win_path_pattern, template):
            template = re.sub(win_path_pattern, r'(.*?)', template)
        mixed_var = r'(?:[\w\.\-\\]*\(\.\*\?\)[\w\.\-\\]*)'
        seg = r'(?:' + mixed_var + r'|[\w\.\-\\]+)'
        p_mid = r'/?(?:' + seg + r'/)+' + mixed_var + r'(?:/' + seg + r')+'
        p_start = mixed_var + r'(?:/' + seg + r'){2,}'
        p_root_start = r'/(?:' + mixed_var + r'/)+' + seg
        p_end_strict = r'(?:' + seg + r'/){2,}' + mixed_var
        p_end_root = r'/(?:' + seg + r'/){1,}' + mixed_var
        br_l = r'(\\?[\[\{\(\<]?)'
        br_r = r'(\\?[\]\}\)\>]?)'
        full_path_pattern = f"{p_mid}|{p_start}|{p_root_start}|{p_end_strict}|{p_end_root}"
        full_path_pattern = br_l + f"({full_path_pattern})" + br_r
        while re.search(full_path_pattern, template):
            template = re.sub(full_path_pattern, r'\1(.*?)\3', template)

        complex_slash_token_pattern = r'(?:[^/\s+\\()]*\(\.\*\?\)[^/\s+\\()]*)/(?:[^/\s+\\()]*\(\.\*\?\)[^/\s+\\()^]*)'
        while re.search(complex_slash_token_pattern, template):
            template = re.sub(complex_slash_token_pattern, r'(.*?)', template)

        while re.search(r'(\\\[|<)\s*(\(\.\*\?\))[a-zA-Z0-9>]+?\s*(\\\]|>)', template):
            template = re.sub(r'(\\\[|<)\s*(\(\.\*\?\))[a-zA-Z0-9>]+?\s*(\\\]|>)', r'\1\2\3', template)

        while re.search(r"(^|\s)/\(\.\*\?\)", template):
            template = re.sub(r"(^|\s)/\(\.\*\?\)", r'\1(.*?)', template)

        while re.search(r"(^|\s)'/\(\.\*\?\)'", template):
            template = re.sub(r"(^|\s)'/\(\.\*\?\)'", r'(.*?)', template)

        while re.search(r'(\(\.\*\?\))\\\+\s*(\(\.\*\?\))', template):
            template = re.sub(r'(\(\.\*\?\))\\\+\s*(\(\.\*\?\))', r'(.*?)', template)

        while re.search(r'(\(\.\*\?\))\\-\\-(\(\.\*\?\))', template):
            template = re.sub(r'(\(\.\*\?\))\\-\\-(\(\.\*\?\))', r'(.*?)', template)

        while re.search(r'(\(\.\*\?\))\\\.\d+\\-(\(\.\*\?\))', template):
            template = re.sub(r'(\(\.\*\?\))\\\.\d+\\-(\(\.\*\?\))', r'(.*?)', template)

        while re.search(r'(\(\.\*\?\))\\\.\d+\\\.(\(\.\*\?\))', template):
            template = re.sub(r'(\(\.\*\?\))\\\.\d+\\\.(\(\.\*\?\))', r'(.*?)', template)

        while re.search(r'(\(\.\*\?\))\\-(\(\.\*\?\))', template):
            template = re.sub(r'(\(\.\*\?\))\\-(\(\.\*\?\))', r'(.*?)', template)

        while re.search(r'(\\\(\(\.\*\?\)\\\))\s*(\\\(\(\.\*\?\)\\\))', template):
            template = re.sub(r'(\\\(\(\.\*\?\)\\\))\s*(\\\(\(\.\*\?\)\\\))', r'\1', template)

        while re.search(r'(\(\.\*\?\))\s*\\=\s*(\(\.\*\?\))', template):
            template = re.sub(r'(\(\.\*\?\))\s*\\=\s*(\(\.\*\?\))', r'(.*?)', template)

        var_pat = r'\(\.\*\?\)'
        sep_pat = r'\s*,\s*'
        list_pattern = var_pat + r'(?:' + sep_pat + var_pat + r'){2,}'
        while re.search(list_pattern, template):
            template = re.sub(list_pattern, r'(.*?)', template)

        while re.search(r'(\(\.\*\?\))\s*,\s*(\(\.\*\?\))\s*,', template):
            template = re.sub(r'(\(\.\*\?\))\s*,\s*(\(\.\*\?\))\s*,', r'(.*?),', template)

        while re.search(r'(\(\.\*\?\)):0\b', template):
            template = re.sub(r'(\(\.\*\?\)):0\b', r'(.*?)', template)

        tokens = re.split('(' + '|'.join(token_delimiters) + ')', template)
        new_tokens = []
        for token in tokens:
            original_token = token
            if re.match(r'^\d+$', token):
                token = '(.*?)'
            elif re.match(r'^0x[0-9a-fA-F]+$', token):
                token = '(.*?)'
            new_tokens.append(token)

        template = ''.join(new_tokens)

        while True:
            prev = template
            template = re.sub(r'\(\.\*\?\)/\(\.\*\?\)', r'(.*?)', template)
            template = re.sub(r'\(\.\*\?\)/([^/]+)/\(\.\*\?\)', r'(.*?)', template)
            if prev == template:
                break

        while True:
            prev = template
            template = re.sub(r'\(\.\*\?\)\\\.\(\.\*\?\)', r'(.*?)', template)
            if prev == template:
                break

        while True:
            prev = template
            template = re.sub(r'\(\.\*\?\)\(\.\*\?\)', '(.*?)', template)
            if prev == template:
                break

        while template.endswith("\\"):
            template = template[:-1]

        while " #(.*?)# " in template:
            template = template.replace(" #(.*?)# ", " (.*?) ")

        while re.search(r'\s(?:\\?#)(\(\.\*\?\))\s', template):
            template = re.sub(r'\s(?:\\?#)(\(\.\*\?\))\s', r' \1 ', template)

        while True:
            prev = template
            template = re.sub(r'(\(\.\*\?\)):\s*(\(\.\*\?\))(?!\s*:)', r'(.*?)', template)
            if prev == template:
                break

        while re.search(r'(\(\.\*\?\))\s+(\(\.\*\?\))', template):
            template = re.sub(r'(\(\.\*\?\))\s+(\(\.\*\?\))', r'(.*?)', template)

        while "(.*?)#(.*?)" in template:
            template = template.replace("(.*?)#(.*?)", "(.*?)")
        while r"(.*?)\#(.*?)" in template:
            template = template.replace(r"(.*?)\#(.*?)", "(.*?)")
        while "(.*?)/(.*?)" in template:
            template = template.replace("(.*?)/(.*?)", "(.*?)")
        while "(.*?)@(.*?)" in template:
            template = template.replace("(.*?)@(.*?)", "(.*?)")
        while "(.*?).(.*?)" in template:
            template = template.replace("(.*?).(.*?)", "(.*?)")
        while ' "(.*?)" ' in template:
            template = template.replace(' "(.*?)" ', ' (.*?) ')
        while " '(.*?)' " in template:
            template = template.replace(" '(.*?)' ", " (.*?) ")
        while "(.*?)(.*?)" in template:
            template = template.replace("(.*?)(.*?)", "(.*?)")

        if log and '(.*?)' in template:
            raw_delims = r'\s\(\)\[\]\{\},\.;\\'
            local_pattern = r'(?:^|(?<=[' + raw_delims + r']))(\(\.\*\?\)):([^' + raw_delims + r':]+)(?=[' + raw_delims + r']|$)'
            try:
                full_regex = template if template.startswith('^') else '^' + template
                full_regex = full_regex if full_regex.endswith('$') else full_regex + '$'
                log_match = re.match(full_regex, log, timeout=2.0)
                if log_match:
                    candidates = list(re.finditer(local_pattern, template))
                    for m in reversed(candidates):
                        group_idx = template[:m.start(1)].count('(.*?)') + 1
                        token1_val = log_match.group(group_idx)
                        if token1_val and token1_val.isalpha():
                            template = template[:m.start(1)] + token1_val + template[m.end(1):]
            except TimeoutError:
                pass
            except Exception as e:
                pass
        return template

    def replace_nth(self, s, old, new, n):
        parts = s.split(old)
        if len(parts) <= n:
            return s
        return old.join(parts[:n]) + new + old.join(parts[n:])

    def check_and_modify_regex(self, regex, string):
        modified_regex = regex
        max_restarts = 4
        restarts = 0

        while restarts < max_restarts:
            analysis_regex = modified_regex.strip()
            if not analysis_regex.startswith('^'):
                analysis_regex = '^' + analysis_regex
            if not analysis_regex.endswith('$'):
                analysis_regex = analysis_regex + '$'

            try:
                pattern = re.compile(analysis_regex)
                match = pattern.search(string)
                if not match:
                    break
                groups = match.groups()
            except:
                break

            changed = False
            for i, group in enumerate(groups, start=1):
                if group is None:
                    continue

                replacement = None

                if group.endswith('.'):
                    if group == '.':
                        continue
                    if re.match(r'^\d+\.\d+\.\d+\.$', group):
                        continue
                    end_pos = match.end(i)
                    next_char = string[end_pos] if end_pos < len(string) else ''
                    if next_char.isalnum() or next_char == '_':
                        continue
                    if next_char in [' ', '[', '(', '{', ':', ';', ''] or end_pos == len(string):
                        replacement = r'(.*?)\.'
                elif re.match(r'^\[.+\]-.+$', group):
                    replacement = r'\[(.*?)\]-(.*?)'
                elif group.startswith('('):
                    if group.endswith(')'):
                        replacement = r'\((.*?)\)'
                    else:
                        replacement = r'\((.*?)'
                elif group.endswith(')'):
                    if group.count(')') > group.count('('):
                        replacement = r'(.*?)\)'
                elif re.fullmatch(r'\*+', group):
                    replacement = '\\' + '\\'.join(list(group))

                if replacement:
                    new_regex = self.replace_nth(modified_regex, '(.*?)', replacement, i)
                    if new_regex != modified_regex:
                        modified_regex = new_regex
                        changed = True
                        break
            if not changed:
                break
            restarts += 1

        return modified_regex.strip()

    def clean_regex(self, log, regex):
        regex = regex.strip()
        if log:
            if re.search(r'[+-]\d+', log):
                regex = re.sub(r'[+-]\s*\(\.\*\?\)', r'(.*?)', regex)
            potential_tokens = re.findall(r'\b[a-zA-Z0-9]+\b', regex)
            unique_tokens = sorted(list(set(potential_tokens)), key=len, reverse=True)

            for token in unique_tokens:
                lower_token = token.lower().strip('.,:')
                if re.fullmatch(r'[a-z]+\d+', lower_token) and len(lower_token) < 4:
                    is_dynamic = False
                    if (token + ':') in log:
                        is_dynamic = True
                    if not is_dynamic:
                        t_idx = log.find(token)
                        if t_idx != -1:
                            pre_text = log[:t_idx].strip().lower()
                            if pre_text:
                                pre_words = re.findall(r'[a-z]+', pre_text)
                                if pre_words and pre_words[-1] in ['on', 'at', 'to']:
                                    is_dynamic = True
                    if is_dynamic:
                        regex = re.sub(rf'\b{re.escape(token)}\b', r'(.*?)', regex)
        if log:
            log_clean_end = log.strip()
            regex_effective_end = re.sub(r'[\s\$]+$', '', regex)
            hallucination_chars = ".;,!?"

            if log_clean_end and regex_effective_end:
                last_log_char = log_clean_end[-1]
                last_reg_char = regex_effective_end[-1]
                if last_reg_char in hallucination_chars and last_reg_char != last_log_char:
                    regex = re.sub(r'\\?' + re.escape(last_reg_char) + r'\s*\$?$', '', regex)
                    regex = regex.rstrip('\\').strip()

            while regex.startswith("'") and not log.strip().startswith("'"):
                regex = regex[1:]
            while regex.endswith("'") and not log.strip().endswith("'"):
                regex = regex[:-1]

            regex = re.sub(r'\d+\\\.\d+', '(.*?)', regex)
            clean_log = log.strip()
            clean_regex_content = regex.lstrip('^').strip()

            if clean_log and clean_regex_content:
                log_char = clean_log[0]
                reg_char = clean_regex_content[0]
                if log_char.isalpha() and reg_char.isalpha():
                    if log_char != reg_char and log_char.lower() == reg_char.lower():
                        first_char_index = regex.find(reg_char)
                        if first_char_index != -1:
                            regex = regex[:first_char_index] + log_char + regex[first_char_index + 1:]

            target_chars = "()[]{};=,:<>*.\"&\\"
            pair_chars = "()[]{}"
            fill_chars = "()[]{}"

            max_iterations = 10
            for _ in range(max_iterations):
                old_regex_iteration = regex
                regex_no_vars = regex.replace(r'(.*?)', '')
                needs_deep_clean = False
                for char in target_chars:
                    if regex_no_vars.count(char) != log.count(char):
                        needs_deep_clean = True
                        break
                if not needs_deep_clean:
                    break

                log_skel_tokens = []
                i = 0
                n_log = len(log)
                while i < n_log:
                    if log[i].isalnum() or log[i] == '_':
                        start = i
                        while i < n_log and (log[i].isalnum() or log[i] == '_'):
                            i += 1
                        log_skel_tokens.append(log[start:i])
                    else:
                        char = log[i]
                        if char == '.' and 0 < i < n_log - 1 and log[i - 1].isdigit() and log[i + 1].isdigit():
                            i += 1
                            continue
                        if char in target_chars:
                            log_skel_tokens.append(char)
                        i += 1

                def get_regex_skel(current_regex):
                    indices, tokens = [], []
                    i, n = 0, len(current_regex)
                    while i < n:
                        if current_regex[i:i + 5] == '(.*?)':
                            i += 5
                            continue

                        if current_regex[i].isalnum() or current_regex[i] == '_':
                            start = i
                            while i < n and (current_regex[i].isalnum() or current_regex[i] == '_'):
                                i += 1
                            tokens.append(current_regex[start:i])
                            indices.append(start)
                        else:
                            char = current_regex[i]
                            if char == '.' and 0 < i < n - 1 and current_regex[i - 1].isdigit() and current_regex[
                                i + 1].isdigit():
                                i += 1
                                continue
                            if char in target_chars:
                                tokens.append(char)
                                indices.append(i)
                            i += 1
                    return indices, tokens

                def get_tokens(text):
                    tok_pat = r'(\\\.|\\\(|\\\)|\\\[|\\\]|\\\:|\\=|\\\+|\\\$|\(\.\*\?\)|[a-zA-Z0-9_]+|\s+|.)'
                    return re.findall(tok_pat, text)

                log_tokens = get_tokens(log)
                reg_tokens = get_tokens(regex)
                reg_tokens_norm = [t.replace("\\", "") if (len(t) == 2 and t.startswith("\\")) else t for t in
                                   reg_tokens]

                matcher = difflib.SequenceMatcher(None, reg_tokens_norm, log_tokens, autojunk=False)
                new_reg_tokens = []

                for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                    if tag == 'equal':
                        new_reg_tokens.extend(reg_tokens[i1:i2])
                    elif tag in ['insert', 'replace']:
                        has_variable = any('(.*?)' in t for t in reg_tokens[i1:i2])
                        if tag == 'replace':
                            new_reg_tokens.extend(reg_tokens[i1:i2])

                        if not has_variable:
                            for k in range(j1, j2):
                                char = log_tokens[k]
                                if char in target_chars and char in fill_chars:
                                    if char in "[]" and "=[]" in log:
                                        continue
                                    escaped_char = "\\" + char if char in "()[]{}.*+?^$|" else char
                                    new_reg_tokens.append(escaped_char)
                    elif tag == 'delete':
                        new_reg_tokens.extend(reg_tokens[i1:i2])

                regex = "".join(new_reg_tokens)

                regex_skel_indices, regex_skel_tokens = get_regex_skel(regex)
                matcher = difflib.SequenceMatcher(None, regex_skel_tokens, log_skel_tokens, autojunk=False)

                indices_to_remove = set()
                regex_metacharacters = "()[]{}.*+?^$|"
                for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                    if tag in ['delete', 'replace']:
                        for k in range(i1, i2):
                            idx = regex_skel_indices[k]
                            token_to_check = regex_skel_tokens[k]

                            if len(token_to_check) == 1 and token_to_check in target_chars:
                                char_to_check = token_to_check
                                if char_to_check == '=':
                                    if log.count('=') >= regex.replace('(.*?)', '').count('='):
                                        continue
                                if regex[idx] == '\\' and idx + 1 < len(regex):
                                    next_char = regex[idx + 1]
                                    if next_char in regex_metacharacters:
                                        continue
                                if char_to_check not in pair_chars:
                                    indices_to_remove.add(idx)

                log_pair_skel = [c for c in log_skel_tokens if len(c) == 1 and c in pair_chars]
                reg_pair_skel_indices = []
                reg_pair_skel_chars = []
                for idx, token in zip(regex_skel_indices, regex_skel_tokens):
                    if len(token) == 1 and token in pair_chars:
                        reg_pair_skel_indices.append(idx)
                        reg_pair_skel_chars.append(token)

                pair_matcher = difflib.SequenceMatcher(None, reg_pair_skel_chars, log_pair_skel, autojunk=False)
                for tag, i1, i2, j1, j2 in pair_matcher.get_opcodes():
                    if tag in ['delete', 'replace']:
                        for k in range(i1, i2):
                            indices_to_remove.add(reg_pair_skel_indices[k])

                if indices_to_remove:
                    regex_list = list(regex)
                    for idx in sorted(list(indices_to_remove), reverse=True):
                        regex_list.pop(idx)
                    regex = "".join(regex_list)

                if regex == old_regex_iteration:
                    break

        regex = regex.replace('\d+\.\d+',
                              '\d+(\.\d+)?') \
            .replace('\\d+', '-?\\d+') \
            .replace('a-f', 'a-z') \
            .replace('A-F', 'A-Z')
        regex = self.correct_single_template(regex, log)
        if log:
            regex = self.generalize_regex(log, regex)
            regex = self.check_and_modify_regex(regex, log)

        if log and "..." in log:
            if " ..." in log:
                if r"\.\.\." in regex and r" \.\.\." not in regex:
                    regex = regex.replace(r"\.\.\.", r" \.\.\.")
                elif "..." in regex and " ..." not in regex:
                    regex = regex.replace("...", " ...")

        regex = regex.replace("<*>", "").replace("()", "")
        return regex

    def generate_log_template_using_pipeline(self,
                                             log_list,
                                             dic=False,
                                             do_sample=False,
                                             max_new_tokens=150,
                                             correction_context=None,
                                             force_llm=False
                                             ):
        if not force_llm and self.check_pre_logs(log_list, dic=dic):
            if dic:
                log_list = get_logs_from_group(log_list)
            return re.escape(log_list[0]).replace("\ ", " "), " "
        try:
            if "chatglm" in self.model:
                messages, sampled_log_list = self.generate_prompt_with_log_list_chatglm(
                    log_list, dic=dic, correction_context=correction_context
                )
                response, history = self.pipeline[0].chat(self.pipeline[1], messages[-1]["content"],
                                                          history=messages[:-1], do_sample=do_sample)
                return self.clean_regex(sampled_log_list[0], self.template_to_regex(response)), response
            elif "qwen" in self.model.lower():
                prompt, sampled_log_list = self.generate_prompt_with_log_list_qwen(
                    log_list, dic=dic, correction_context=correction_context
                )

                terminators = [self.tokenizer.eos_token_id]
                vocab = self.tokenizer.get_vocab()
                if "<|im_end|>" in vocab:
                    terminators.append(self.tokenizer.convert_tokens_to_ids("<|im_end|>"))
                if "<|endoftext|>" in vocab:
                    terminators.append(self.tokenizer.convert_tokens_to_ids("<|endoftext|>"))
                terminators = [t for t in terminators if t is not None]
                safe_pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 151643

                outputs = self.pipeline(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    eos_token_id=terminators,
                    pad_token_id=safe_pad_token_id
                )
                raw_out = outputs[0]["generated_text"][len(prompt):]
                raw_out = self._abstract_dynamic_structural_tokens(raw_out, log_list)
                return self.clean_regex(sampled_log_list[0], self.template_to_regex(raw_out)), raw_out
            elif "Mistral" in self.model or "codegemma" in self.model:
                prompt, sampled_log_list = self.generate_prompt_with_log_list_mistral(
                    log_list, dic=dic, correction_context=correction_context
                )
                outputs = self.pipeline(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                )
                raw_out = outputs[0]["generated_text"][len(prompt):]
                return self.clean_regex(sampled_log_list[0],
                                        self.template_to_regex(raw_out)), raw_out
            else:
                prompt, sampled_log_list = self.generate_prompt_with_log_list(
                    log_list, dic=dic, correction_context=correction_context
                )
                terminators = [
                    self.tokenizer.eos_token_id,
                    self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                ]
                sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=max_new_tokens,
                    stop_token_ids=terminators
                )
                outputs = self.pipeline.generate([prompt], sampling_params, use_tqdm=False)
                out = outputs[0].outputs[0].text
                out = self._abstract_dynamic_structural_tokens(out, log_list)
                resul = self.clean_regex(sampled_log_list[0], self.template_to_regex(out))
                return resul, out
        except torch.cuda.OutOfMemoryError:
            print(f"Out of memory, try to reduce the number of samples", flush=True)
            self.regex_sample = self.regex_sample - 1
            if "chatglm" in self.model:
                messages, sampled_log_list = self.generate_prompt_with_log_list_chatglm(
                    log_list, dic=dic, correction_context=correction_context
                )
                response, history = self.pipeline[0].chat(self.pipeline[1], messages[-1]["content"],
                                                          history=messages[:-1], do_sample=do_sample)
                self.regex_sample = self.regex_sample + 1
                return self.clean_regex(sampled_log_list[0], self.template_to_regex(response)), response
            elif "Mistral" in self.model or "codegemma" in self.model:
                prompt, sampled_log_list = self.generate_prompt_with_log_list_mistral(
                    log_list, dic=dic, correction_context=correction_context
                )
                outputs = self.pipeline(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                )
                self.regex_sample = self.regex_sample + 1
                raw_out = outputs[0]["generated_text"][len(prompt):]
                return self.clean_regex(sampled_log_list[0],
                                        self.template_to_regex(outputs[0]["generated_text"][len(prompt):])), raw_out
            elif "qwen" in self.model.lower():
                prompt, sampled_log_list = self.generate_prompt_with_log_list_qwen(
                    log_list, dic=dic, correction_context=correction_context
                )

                terminators = [self.tokenizer.eos_token_id]
                vocab = self.tokenizer.get_vocab()
                if "<|im_end|>" in vocab:
                    terminators.append(self.tokenizer.convert_tokens_to_ids("<|im_end|>"))
                if "<|endoftext|>" in vocab:
                    terminators.append(self.tokenizer.convert_tokens_to_ids("<|endoftext|>"))
                terminators = [t for t in terminators if t is not None]
                safe_pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 151643

                outputs = self.pipeline(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    eos_token_id=terminators,
                    pad_token_id=safe_pad_token_id
                )
                self.regex_sample = self.regex_sample + 1
                raw_out = outputs[0]["generated_text"][len(prompt):]
                return self.clean_regex(sampled_log_list[0], self.template_to_regex(raw_out)), raw_out
            else:
                prompt, sampled_log_list = self.generate_prompt_with_log_list(
                    log_list, dic=dic, correction_context=correction_context
                )
                terminators = [
                    self.tokenizer.eos_token_id,
                    self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                ]

                sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=max_new_tokens,
                    stop_token_ids=terminators
                )
                outputs = self.pipeline.generate([prompt], sampling_params, use_tqdm=False)
                out = outputs[0].outputs[0].text
                self.regex_sample = self.regex_sample + 1
                return self.clean_regex(sampled_log_list[0], self.template_to_regex(out)), out

    def get_logs_from_group(self, group_list):
        logs_from_group = []
        for ele in group_list:
            logs_from_group.append(ele["Content"])
        return logs_from_group

    def find_longest_backtick_content(self, text):
        is_multiline = '\n' in text
        if is_multiline:
            matches = re.findall(r"`(.*?)`", text)
            if not matches or matches == [""]:
                matches = re.findall(r"`(.*)", text)
            if matches:
                return max(matches, key=len)
        return text

    def _get_proactive_instructions(self, sample_logs):
        instructions = []

        has_suspicious_kv = False
        for log in sample_logs:
            if self._detect_parameters_in_raw_log(log):
                has_suspicious_kv = True
                break

        if has_suspicious_kv:
            instructions.append(
                "Detected 'key=value' patterns. You MUST abstract the value part as a variable "
                "(e.g., 'uid=123' -> 'uid=<*>', 'ip=1.2.3.4' -> 'ip=<*>') ."
            )

        combined_log_text = " ".join(sample_logs)
        has_path = re.search(r'(?:^|[\s:=\'"\[\(])(?:/|https?://|www\.)[a-zA-Z0-9._/-]+', combined_log_text)

        if has_path:
            instructions.append(
                "Detected file paths (starting with '/'). You should generally abstract specific file paths as variables (e.g., '/home/user/doc' -> '<*>')."
            )

        if instructions:
            return " " + " ".join(instructions)
        return ""

    def clean_generated_regex(self, log, template):
        template = self.find_longest_backtick_content(template)
        template = template.strip()

        template = re.sub(r'\\+(\()', r'\1', template)
        template = re.sub(r'\\+(\))', r'\1', template)
        template = re.sub(r'\\+(\.)', r'\1', template)

        while template.endswith("$"):
            template = template[:-1].strip()
        while template.startswith("`"):
            template = template[1:]
        while template.startswith('"'):
            template = template[1:]
        while template.startswith("^"):
            template = template[1:]
        while template.startswith("\b"):
            template = template[2:]

        while template.endswith("`"):
            template = template[:-1]
        while template.endswith('"'):
            template = template[:-1]
        while template.endswith("$"):
            template = template[:-1]
        while template.endswith("\b"):
            template = template[:-2]
        while template.endswith("finished"):
            template = template[:-8]
        template = template.replace(",", "")
        template = template.replace("\ ", " ")

        return self.clean_regex(log=log, regex=template)

    def _detect_parameters_in_raw_log(self, log_content):
        param_pattern = r'\b(\w+)\s*=\s*([\w./:-]+)'

        matches = re.findall(param_pattern, log_content)

        suspicious_params = []
        for key, value in matches:
            if key.isdigit():
                continue

            if len(value) > 0:
                suspicious_params.append(f"{key}={value}")

        return suspicious_params

    def detect_hardcoded_parameters(self, template_regex):
        clean_template = template_regex.replace(r'(.*?)', '<VAR>')
        clean_template = clean_template.replace('\\', '')
        hardcoded_pattern = r'\b(\w+)\s*=\s*([\w./:-]+)(?!\s*=)'
        matches = re.findall(hardcoded_pattern, clean_template)

        suspicious_params = []
        for key, value in matches:
            if key.isdigit():
                continue
            if value != '<VAR>':
                suspicious_params.append(f"{key}={value}")

        colon_list_pattern = r'\b(\w+)\s*:\s*([^,\s]+)\s*,\s*\b(\w+)\s*:\s*([^,\s\)\]]+)'
        colon_matches = re.findall(colon_list_pattern, clean_template)

        for k1, v1, k2, v2 in colon_matches:
            if v1 != '<VAR>':
                suspicious_params.append(f"{k1}:{v1}")
            if v2 != '<VAR>':
                suspicious_params.append(f"{k2}:{v2}")

        return suspicious_params

    def check_regex_from_groups(self, res_list, groups_dict_list, log_regex, new_event=0, history_str=None):
        if history_str is None:
            history_str = log_regex
        wrong_logs = []

        cleaned_candidate = self.regex_manager1.apply_cleaning_rules(log_regex)

        has_hardcoded_errors = False
        hardcoded_errors_msg = None

        if self.do_self_reflection == "True" and cleaned_candidate:
            if '=' in cleaned_candidate or ':' in cleaned_candidate:
                hardcoded_errors = self.detect_hardcoded_parameters(cleaned_candidate)
                if hardcoded_errors:
                    has_hardcoded_errors = True
                    hardcoded_errors_msg = hardcoded_errors

        unique_content_cache = {}
        fast_mode = False

        if cleaned_candidate and not has_hardcoded_errors:
            all_unique_contents = list({log["Content"] for log in groups_dict_list})
            sample_size = min(30, len(all_unique_contents))
            sampled_contents = random.sample(all_unique_contents, sample_size)

            passed_count = 0
            for content in sampled_contents:
                is_match = verify_one_regex(content, cleaned_candidate)
                unique_content_cache[content] = is_match
                if is_match:
                    passed_count += 1
                else:
                    break

            if passed_count == sample_size and sample_size > 0:
                fast_mode = True

        for log in groups_dict_list:
            line_id = log.get("LineId", "")
            content = log["Content"]

            if not cleaned_candidate:
                is_match = False
            else:
                if has_hardcoded_errors:
                    is_match = False
                    if content not in unique_content_cache:
                        unique_content_cache[content] = False
                else:
                    if fast_mode:
                        is_match = verify_one_regex(content, cleaned_candidate)
                    else:
                        if content in unique_content_cache:
                            is_match = unique_content_cache[content]
                        else:
                            is_match = verify_one_regex(content, cleaned_candidate)
                            unique_content_cache[content] = is_match

            if is_match:
                current_log_regex = cleaned_candidate
                if self.do_self_reflection == "True":
                    cleaned_regex = self.regex_manager1.add_regex_template(current_log_regex, content)
                    if cleaned_regex:
                        current_log_regex = cleaned_regex
                    if new_event == 0:
                        res_list.append((line_id, content, log["EventId"], current_log_regex, history_str))
                    else:
                        res_list.append((line_id, content, new_event, current_log_regex, history_str))
                else:
                    if verify_one_regex(content, current_log_regex):
                        self.regex_manager1.add_regex_template(current_log_regex, content)
                    res_list.append((line_id, content, log["EventId"], current_log_regex, history_str))
            else:
                wrong_logs.append(log)

        return res_list, wrong_logs

    def store_regx_for_logs(self, res_list, groups_dict_list, log_regex, initial_raw_output=""):
        history_list = [f"Attempt 1 Llama Output: {initial_raw_output}"]
        history_str = " \n".join(history_list)
        previous_regex = log_regex
        repeat_count = 0
        res_list, wrong_logs = self.check_regex_from_groups(
            res_list, groups_dict_list, log_regex, history_str=history_str
        )
        len_wrong = len(wrong_logs)
        test_time = 0
        max_retries = 3
        while len(wrong_logs) > 0 and (test_time < max_retries and len_wrong == len(wrong_logs)):
            len_wrong = len(wrong_logs)
            test_time += 1

            failed_log = wrong_logs[0]["Content"] if wrong_logs else ""

            correction_context = self._generate_correction_context(
                original_template=log_regex,
                failed_log=failed_log
            )

            log_regex, raw_output = self.generate_log_template_using_pipeline(
                log_list=wrong_logs,
                dic=True,
                correction_context=correction_context
            )

            if log_regex == previous_regex:
                repeat_count += 1
                print(f"[Early Stop] Duplicate template generated at attempt {test_time + 1}. Stopping recursion.")
                if repeat_count >= 1:
                    break
            else:
                repeat_count = 0
                previous_regex = log_regex

            history_list.append(f"Attempt {test_time + 1} Llama Output: {raw_output}")
            history_str = " \n".join(history_list)

            res_list, wrong_logs = self.check_regex_from_groups(
                res_list, wrong_logs, log_regex, self.new_event, history_str=history_str
            )
            if len_wrong != len(wrong_logs):
                self.new_event += 1

        for log in wrong_logs:
            res_list.append((log["LineId"], log["Content"], str(self.new_event), log_regex, history_str))
            self.new_event += 1
            self.unmatched_logs.append((log, log_regex))

        return res_list

    def remove_first_matching_item(self, data, content_to_remove):
        for i, item in enumerate(data):
            if item['Content'] == content_to_remove:
                del data[i]
                break
        return data

    def parse(self, groups_from_parser, logs):
        res_list = []
        start_time = datetime.now()
        original_log_count = len(logs)

        if not logs:
            return res_list

        if self.check_pre_logs(log_list=logs, dic=True):
            res_list = self.store_regx_for_logs(
                res_list,
                groups_from_parser,
                re.escape(logs[0]["Content"]).replace("\ ", " "),
                initial_raw_output="[Pre-check] Static Log, Skipped LLM"
            )
        else:
            check_sample_size = min(len(logs), 100)
            representative_samples = random.sample(logs, check_sample_size)

            any_sample_matched = False
            first_matched_regex = None
            for s_log in representative_samples:
                matched_regex = self.regex_manager1.find_matched_regex_template(s_log["Content"])
                if matched_regex:
                    any_sample_matched = True
                    first_matched_regex = matched_regex
                    break

            matched_logs = []
            if any_sample_matched:
                last_matched_regex = first_matched_regex

                for log in logs.copy():
                    content = log["Content"]

                    if last_matched_regex and verify_one_regex(content, last_matched_regex):
                        matched_regex = last_matched_regex
                    else:
                        matched_regex = self.regex_manager1.find_matched_regex_template(content)

                    if matched_regex:
                        matched_logs.append(log)
                        res_list.append((log["LineId"], content, "0", matched_regex,
                                         "Matched Cache: " + matched_regex))
                        last_matched_regex = matched_regex
            else:
                print(f"DEBUG: 组预检失败（100条采样均未命中），跳过 {len(logs)} 条日志的缓存匹配，直接进入 LLM 解析。")

            if matched_logs:
                matched_line_ids = {log["LineId"] for log in matched_logs}
                logs = [log for log in logs if log["LineId"] not in matched_line_ids]
                groups_from_parser = [g for g in groups_from_parser if g.get("LineId") not in matched_line_ids]

            if logs:
                log_regex, raw_out = self.generate_log_template_using_pipeline(
                    log_list=logs,
                    dic=True
                )
                res_list = self.store_regx_for_logs(
                    res_list,
                    groups_from_parser,
                    log_regex,
                    initial_raw_output=raw_out
                )

        time_taken = datetime.now() - start_time
        self.total_time += time_taken.total_seconds()

        if len(res_list) != original_log_count:
            print(f"警告：解析结果行数（{len(res_list)}）与输入日志数（{original_log_count}）不匹配！")

        return res_list

    def print_time(self):
        print("[LLaMa parsing time taken: {!s}]".format(self.total_time), flush=True)