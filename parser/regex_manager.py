import regex as re
import string
from datetime import datetime


def verify_one_regex_to_match_whole_log(log, regex):
    log_clean = log.replace(",", "")
    regex_clean = regex.replace(",", "")
    template_equals_count = regex_clean.count('=')
    log_clean1 = re.sub(r'\([^\)]*\)', '', log_clean)
    log_outside_equals = re.sub(r'\{[^\}]*\}', '', log_clean1).count('=')
    log_total_equals = log_clean.count('=')
    colon_indices = [m.start() for m in re.finditer(':', log_clean)]
    log_truncated_equals = -1

    if len(colon_indices) >= 2:

        log_before_second_colon = log_clean[:colon_indices[1]]
        log_truncated_equals = log_before_second_colon.count('=')

    if (template_equals_count != log_total_equals and
            template_equals_count != log_outside_equals and
            template_equals_count != log_truncated_equals):
        return False

    if not regex_clean.startswith('^'):
        regex_clean = '^' + regex_clean
    if not regex_clean.endswith('$'):
        regex_clean = regex_clean + '$'

    try:
        match = re.search(regex_clean, log_clean, timeout=2.0)
        if not match:
            return False


        groups = match.groups()
        if len(groups) > 1:

            for i in range(len(groups) - 1):
                group_content = groups[i]

                if group_content and '=' in group_content:

                    return False

        return True
    except (re.error, TimeoutError):
        return False


def is_punctuation_or_space(s):
    allowed_chars = string.punctuation + ' '
    filtered_string = ''.join(char for char in s if char not in allowed_chars)
    return all(char in allowed_chars for char in s) or len(filtered_string) < 3


class RegexTemplateManager:
    def __init__(self):
        self.templates = []
        self.regex_template_set = set()
        self.total_time = 0.0


    def apply_cleaning_rules(self, regex_template):
        regex_template = re.sub(r"<\s*\*\s*>", "(.*?)", regex_template)
        if is_punctuation_or_space(regex_template):
            #print(f"DEBUG: apply_cleaning_rules REJECTED: {regex_template}")
            return False
        regex_template = re.sub(r"\bfrom\s+\S+\s+to\s+\S+", "from (.*?) to (.*?)", regex_template)
        max_iterations = 2
        for _ in range(max_iterations):
            original = regex_template



            regex_template = re.sub(r"\(\.\*\?\)+://\(\.\*\?\)", "(.*?)", regex_template)

            regex_template = re.sub(r"www\.\(\.\*\?\)", "(.*?)", regex_template)



            if regex_template.startswith("_") and regex_template.startswith("_(.*?)"):
                regex_template = regex_template[1:]

            
            if regex_template == original:
                break
        return regex_template.strip()

    def _count_static_tokens(self, template):


        static_content = template.replace("(.*?)", " ")

        tokens = re.findall(r'\b[a-zA-Z0-9_]{2,}\b', static_content)
        return len(tokens)

    def add_regex_template(self, regex_template, log=False):
        regex_template = self.apply_cleaning_rules(regex_template)
        if not regex_template or regex_template in self.regex_template_set:
            return regex_template

        if log and not verify_one_regex_to_match_whole_log(log, regex_template):
            return None

        self.regex_template_set.add(regex_template)
        static_token_count = self._count_static_tokens(regex_template)
        if static_token_count <= 1:

            # print(f"DEBUG: Template '{regex_template}' has too few static tokens ({static_token_count}). Skipped from candidate pool.")

            return regex_template

        word_count = regex_template.count(' ') + 1

        self.templates.append((word_count, regex_template))
        self.templates.sort(key=lambda x: x[0], reverse=True)

        return regex_template

    def add_regex_templates(self, regex_templates):
        for regex_template in regex_templates:
            self.add_regex_template(regex_template)

    def print_regex_templates(self):

        print("\n--- Current Regex Templates in Manager (Global List) ---", flush=True)
        for word_count, regex_template in self.templates:
            print(f"Word Count: {word_count}, Regex: {regex_template}", flush=True)

    def get_index_by_length(self, max_length):
        left, right = 0, len(self.templates) - 1
        target_index = len(self.templates)

        while left <= right:
            mid = (left + right) // 2
            if self.templates[mid][0] <= max_length:
                target_index = mid  
                right = mid - 1
            else:
                left = mid + 1

        return target_index

    def get_regex_templates_by_length(self, log, max_length):


        return [t for t in self.templates if max_length - 8 <= t[0] <= max_length + 1]

    def find_matched_regex_template(self, log):

        start_time = datetime.now()

        if not self.templates:
            self.total_time += (datetime.now() - start_time).total_seconds()
            return False

        log_length = len(log.split())


        start_index = self.get_index_by_length(log_length)


        if start_index == -1 or start_index >= len(self.templates):
            return None


        for i in range(start_index, len(self.templates)):

            _, regex = self.templates[i]


            if verify_one_regex_to_match_whole_log(log, regex):
                return regex

        self.total_time += (datetime.now() - start_time).total_seconds()
        return False



    def print_time(self):
        print("[Regex matching time taken: {!s}]".format(self.total_time), flush=True)


if __name__ == '__main__':
    manager = RegexTemplateManager()
    manager.add_regex_template("jk2_init\(\) Found child (.*?) in scoreboard slot (.*?)$")
    manager.add_regex_template("workerEnv\.init\(\) ok (.*?)$")
    manager.add_regex_template("mod_jk child workerEnv in error state (.*?)$")
    manager.print_regex_templates()
    a = manager.get_regex_templates_by_length(6)
    print(a)
    x = manager.find_matched_regex_template("mod_jk child workerEnv in error sta1te 6")
    print(x)
