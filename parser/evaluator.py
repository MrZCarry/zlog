import matplotlib.pyplot as plt
import seaborn as sns
import grouping
import warnings
import pickle
warnings.filterwarnings("ignore", message=".*do_sample.*")
import os
import torch
import csv
import sys
import argparse
import llama_parser
import transformers
import regex as re
import gc
import pandas as pd
import regex_manager
import accuracy
from transformers import AutoTokenizer, AutoModel
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from auto_marker_generator import UnsupervisedMarkerExtractor

parser = argparse.ArgumentParser()
parser.add_argument("--project", type=str, default="Apache")
parser.add_argument(
    "--model",
    type=str,
    default="../models/Meta-Llama-3-8B-Instruct",
)
parser.add_argument("--sample", type=str, default="5")
parser.add_argument("--similarity", type=str, default="jaccard")
parser.add_argument("--do_self_reflection", type=str, default="True")
args = parser.parse_args()

datasets_full = args.project.split(",")
model_path = args.model
similarity = args.similarity
regex_sample = int(args.sample)
do_self_reflection = args.do_self_reflection

benchmark_settings = {
    "HDFS": {
        "log_file": "HDFS/HDFS_2k.log",
        "log_format": "<Date> <Time> <Pid> <Level> <Component>: <Content>",
        "regex": [r"blk_-?\d+", r"(\d+\.){3}\d+(:\d+)?"],
        "depth":1,
        "st":0.5
    },
    "Hadoop": {
        "log_file": "Hadoop/Hadoop_2k.log",
        "log_format": "<Date> <Time> <Level> \[<Process>\] <Component>: <Content>",
        "regex": [
            r"(\d+\.){3}\d+",       
            r"\d{2}:\d{2}:\d{2}",   
            r"\d+"                  
],
        "depth":1,
        "st":0.5
    },
    "Spark": {
        "log_file": "Spark/Spark_2k.log",
        "log_format": "<Date> <Time> <Level> <Component>: <Content>",
        "regex": [r"(\d+\.){3}\d+", r"\b[KGTM]?B\b", r"([\w-]+\.){2,}[\w-]+", r"\d+"],
        "depth":1,
        "st":0.5
    },
    "Zookeeper": {
        "log_file": "Zookeeper/Zookeeper_2k.log",
        "log_format": "<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>",
        "regex": [r"(/|)(\d+\.){3}\d+(:\d+)?", r"0x[a-f0-9A-F]+", r"\d+"],
        "depth":3,
        "st":0.5
    },
    "BGL": {
        "log_file": "BGL/BGL_2k.log",
        "log_format": "<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>",
        "regex": [r"core\.\d+", r"(\d+\.){3}\d+", r"0x[a-f0-9A-F]+", r"\d+"],
        "depth":1,
        "st":0.5
    },
    "HPC": {
        "log_file": "HPC/HPC_2k.log",
        "log_format": "<LogId> <Node> <Component> <State> <Time> <Flag> <Content>",
        "regex": [r"=\d+", r"0x[a-f0-9A-F]+", r"\d+"],
        "depth":1,
        "st":0.5
    },
    "Thunderbird": {
        "log_file": "Thunderbird/Thunderbird_2k.log",
        "log_format": "<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>",
        "regex": [
            r"(\d+\.){3}\d+",                
            r"\/[\w\.\/-]+",                
            r"0x[a-f0-9A-F]+",               
            r"\b[0-9a-fA-F]{8,}\b",          
            r"(\d+)"                         
        ],
        "depth":1,
        "st":0.5
    },
    "Windows": {
        "log_file": "Windows/Windows_2k.log",
        "log_format": "<Date> <Time>, <Level>                  <Component>    <Content>",
        "regex": [r"0x.*?\s"],
        "depth":3,
        "st":0.5
    },
    "Linux": {
        "log_file": "Linux/Linux_2k.log",
        "log_format": "<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>",
        "regex": [
            r"(\d+\.){3}\d+",      
            r"\d{2}:\d{2}:\d{2}",  
            r"\/[\w\.\/-]+",                
            r"\[\s*\d+\.\d+\s*\]",          
            r"((?<=\s)|(?<=^))0x[a-f0-9A-F]+",
            r"\b[0-9a-fA-F]{10,}\b",
            r"\d+",
        ],
        "depth":1,
        "st":0.5
    },
    "Android": {
        "log_file": "Android/Android_2k.log",
        "log_format": "<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>",
        "regex": [
            r"(/[\w-]+)+",
            r"([\w-]+\.){2,}[\w-]+",
            r"\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b"],
        "depth":3,
        "st":0.5
        ,
    },
    "HealthApp": {
        "log_file": "HealthApp/HealthApp_2k.log",
        "log_format": "<Time>\|<Component>\|<Pid>\|<Content>",
        "regex": [r"([\w-]+\.){2,}[\w-]+", r"\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b"],
        "depth":1,
        "st":0.5
    },
    "Apache": {
        "log_file": "Apache/Apache_full.log",
        "log_format": "\[<Time>\] \[<Level>\] <Content>",
        "regex": [r"(\d+\.){3}\d+", r"\d+"],
        "depth":3,
        "st":0.5
    },
    "Proxifier": {
        "log_file": "Proxifier/Proxifier_2k.log",
        "log_format": "\[<Time>\] <Program> - <Content>",
        "regex": [
            r"<\d+\ssec",                             
            r"([\w-]+\.)+[\w-]+(:\d+)?",              
            r"\d{2}:\d{2}(:\d{2})*",                  
            r"[KGTM]B",                               
        ],
        "depth":2,
        "st": 0.5
        ,
    },
    "OpenSSH": {
        "log_file": "OpenSSH/OpenSSH_2k.log",
        "log_format": "<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>",
        "regex": [r"(\d+\.){3}\d+", r"([\w-]+\.){2,}[\w-]+", r"\d+"],
        "depth":2,
        "st":0.5
    },
    "OpenStack": {
        "log_file": "OpenStack/OpenStack_2k.log",
        "log_format": "<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>",
        "regex": [
            r"((\d+\.){3}\d+,?)+",
            r"[a-fA-F0-9]{8}(?:-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12}",
            r"req-[a-fA-F0-9-]+",
            r"/.+?\s",
            r"\d+"
        ],
        "depth":1,
        "st":0.5
    },
    "Mac": {
        "log_file": "Mac/Mac_2k.log",
        "log_format": "<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>",
        "regex": [r"([\w-]+\.){2,}[\w-]+", r"\d+"],
        "depth":1,
        "st":0.5
    },
    "TestHadoop": {
        "log_file": "Test/Test_2k.log",
        "log_format": "<Date> <Time> <Level> \[<Process>\] <Component>: <Content>",
        "regex": [
            r"(\d+\.){3}\d+",       
            r"\d{2}:\d{2}:\d{2}",   
            r"\d+"                  
],
        "depth":3,
        "st":0.5
    },
    "TestLinux": {
        "log_file": "Linux/Linux_2k.log",
        "log_format": "<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>",
        "regex": [
            r"(\d+\.){3}\d+",      
            r"\d{2}:\d{2}:\d{2}",  
            r"\/[\w\.\/-]+",                
            r"\[\s*\d+\.\d+\s*\]",          
            r"((?<=\s)|(?<=^))0x[a-f0-9A-F]+",
            r"\b[0-9a-fA-F]{10,}\b",
            r"\d+",
        ],
        "depth":3,
        "st":0.5,
},
    "TestOpenStack": {
        "log_file": "OpenStack/OpenStack_2k.log",
        "log_format": "<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>",
        "regex": [
            r"((\d+\.){3}\d+,?)+",
            r"[a-fA-F0-9]{8}(?:-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12}",
            r"req-[a-fA-F0-9-]+",
            r"/.+?\s",
            r"\d+"
        ],
        "depth":3,
        "st":0.5
},
    "Testxieyi": {
        "log_file": "Linux/Linux_2k.log",
        "log_format": "<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>",
        "regex": [
            r"(\d+\.){3}\d+",      
            r"\d{2}:\d{2}:\d{2}",  
            r"\/[\w\.\/-]+",                
            r"\[\s*\d+\.\d+\s*\]",          
            r"((?<=\s)|(?<=^))0x[a-f0-9A-F]+",
            r"\b[0-9a-fA-F]{10,}\b",
            r"\d+",
        ],
        "depth":3,
        "st":0.5,
},
    "TestBGL": {
        "log_file": "BGL/BGL_2k.log",
        "log_format": "<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>",
        "regex": [r"core\.\d+", r"(\d+\.){3}\d+", r"0x[a-f0-9A-F]+", r"\d+"],
        "depth":3,
        "st":0.5
    },

    "TestSpark": {
        "log_file": "Spark/Spark_2k.log",
        "log_format": "<Date> <Time> <Level> <Component>: <Content>",
        "regex": [r"(\d+\.){3}\d+", r"\b[KGTM]?B\b", r"([\w-]+\.){2,}[\w-]+", r"\d+"],
        "depth":3,
        "st":0.5
    },

    "TestMac": {
        "log_file": "Mac/Mac_2k.log",
        "log_format": "<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>",
        "regex": [r"([\w-]+\.){2,}[\w-]+", r"\d+"],
        "depth":3,
        "st":0.5
    },
}


def sort_csv_by_lineid(csv_path):
    """Sort CSV file by LineId in ascending order (global sort)"""
    if not os.path.exists(csv_path):
        print(f"File does not exist: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    if 'LineId' not in df.columns:
        print("Missing LineId column in CSV file, cannot sort")
        return

    try:
        df['LineId'] = df['LineId'].astype(int)
    except ValueError:
        print("LineId contains non-numeric values, cannot convert to integer")
        return

    df_sorted = df.sort_values(by='LineId', ascending=True).reset_index(drop=True)

    df_sorted.to_csv(csv_path, index=False)
    print(f"Globally sorted {csv_path} by LineId, total {len(df_sorted)} lines")


def log_file_to_logs(
    log_file, logformat, first_lines_percent=100, start_line_percent=0, max_lines=None
):
    """Function to transform log file to dataframe, reads from a specific start line and reads up to a given percent of lines."""
    headers, regex = generate_logformat_regex(logformat)
    log_messages = []
    with open(log_file, "r") as fin:
        lines = fin.readlines()
        total_lines = len(lines)  
        start_line = int(
            total_lines * start_line_percent / 100
        )  
        lines_to_read = int(
            (total_lines - start_line) * (first_lines_percent / 100)
        )

        if max_lines is not None:
            lines_to_read = min(lines_to_read, max_lines)

        end_line = min(start_line + lines_to_read, total_lines)
        actual_lines_to_read = end_line - start_line  
        skipped_lines = 0  

        for i, line in enumerate(
            lines[start_line:end_line], start=start_line
        ):

            try:
                match = regex.search(line.strip())
                if match:
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                else:
                    skipped_lines += 1  
                    print(f"Line {i + 1} does not match log format, skipped: {line.strip()}")
            except Exception as e:
                print("Skip line: ", line)
                print(f"Line {i + 1} parsing error, skipped: {line.strip()}, error: {e}")
                skipped_lines += 1  


    logdf = pd.DataFrame(log_messages, columns=headers)
    logdf.insert(
        0, "LineId", range(start_line + 1, start_line + len(log_messages) + 1)
    )  

    array_result = logdf.loc[:, ["Content"]].values
    list_result = [list(row) for row in array_result]

    print(f"log_file_to_logs output lines: {len(list_result)}", flush=True)

    return list_result

def evaluate_roberta_markers(system, records, ground_truth_file, out_csv="roberta_marker_stats.csv"):
    if not records:
        return

    try:
        import pandas as pd
        df_gt = pd.read_csv(ground_truth_file, usecols=["Content", "EventTemplate"])
        df_gt = df_gt.drop_duplicates(subset=["Content"])
        gt_mapping = dict(zip(df_gt['Content'], df_gt['EventTemplate']))
    except Exception as e:
        print(f"Failed to load Ground Truth, skipping RoBERTa stats: {e}")
        return

    total_scored = len(records)
    total_passed = 0
    correct_passed = 0
    plot_data = []

    for r in records:
        score = r.get('score', 0.0)
        token = str(r['token'])
        
        if r['passed']:
            total_passed += 1
            
        gt_template = str(gt_mapping.get(r['log_content'], ""))
        clean_template = gt_template.replace('<*>', ' ').replace('(.*?)', ' ')
        
        import re
        is_static = False
        if re.search(r'(?<![a-zA-Z0-9_])' + re.escape(token) + r'(?![a-zA-Z0-9_])', clean_template):
            is_static = True
        elif token in clean_template and not token.isalnum():
            is_static = True

        if r['passed'] and is_static:
            correct_passed += 1
            
        label = "True Static" if is_static else "True Dynamic"
        
        plot_data.append({
            "Dataset": system,
            "Token": token,
            "Raw_Score": score if score > 0 else 1e-6, 
            "Label": label,
            "Decision": "KEEP (Passed)" if r['passed'] else "DROP (Failed)"
        })

    import os
    file_exists = os.path.isfile(out_csv)
    with open(out_csv, 'a', encoding='utf-8') as f:
        if not file_exists:
            f.write("Dataset,Total_Scored,Total_Passed,Correct_Passed\n")
        f.write(f"{system},{total_scored},{total_passed},{correct_passed}\n")

    print(f"[{system}] RoBERTa decision stats -> Total words: {total_scored}, KEEP count: {total_passed}, Correct KEEP count: {correct_passed}", flush=True)

    if plot_data:
        df_plot = pd.DataFrame(plot_data)
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set_theme(style="whitegrid")
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            sns.stripplot(
                data=df_plot, x="Label", y="Raw_Score", hue="Decision",
                palette={"KEEP (Passed)": "#2ecc71", "DROP (Failed)": "#e74c3c"},
                jitter=True, alpha=0.7, size=5, ax=axes[0]
            )
            axes[0].set_yscale('log')
            axes[0].axhline(y=0.1, color='black', linestyle='--', linewidth=2, label='Base Threshold = 0.1')
            axes[0].set_title(f"Raw Predictability Score Scatter - {system}")
            axes[0].set_ylabel("Raw RoBERTa Score (Log Scale)")
            axes[0].legend(loc='lower right')

            sns.countplot(
                data=df_plot, x="Label", hue="Decision",
                palette={"KEEP (Passed)": "#2ecc71", "DROP (Failed)": "#e74c3c"},
                ax=axes[1]
            )
            axes[1].set_title(f"Final Classification Results - {system}")
            axes[1].set_ylabel("Count of Tokens")
            
            for p in axes[1].patches:
                height = p.get_height()
                if not pd.isna(height) and height > 0:
                    axes[1].annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                                     ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 2),
                                     textcoords='offset points')

            plt.tight_layout()
            img_path = out_csv.replace(".csv", f"_{system}_raw_score_scatter.png")
            plt.savefig(img_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📊 [{system}] Raw score scatter plot generated: {img_path}")
            
        except Exception as e:
            print(f"Failed to plot: {e}")
def read_column_from_csv(file_path, column_name="Content"):
    column_data = []
    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if column_name in row:
                column_data.append(row[column_name])
            else:
                raise ValueError(
                    f"The column '{column_name}' does not exist in the CSV file."
                )
    return column_data

def generate_logformat_regex(logformat):
    """
    Function to generate regular expression to split log messages

    """
    headers = []
    splitters = re.split(r"(<[^<>]+>)", logformat)
    regex = ""
    for k in range(len(splitters)):
        if k % 2 == 0:
            splitter = re.sub(" +", r"\s+", splitters[k])
            regex += splitter
        else:
            header = splitters[k].strip("<").strip(">")
            regex += "(?P<%s>.*?)" % header
            headers.append(header)
    regex = re.compile("^" + regex + "$")
    return headers, regex


def group_logs_using_parser(grouped_logs):
    df = pd.DataFrame(grouped_logs, columns=["LineId", "Content", "EventId", "EventTemplate"])
    df = df[["LineId", "Content", "EventId", "EventTemplate"]]
    grouped = df.groupby("EventId")
    groups_dict = {}
    for name, group in grouped:
        groups_dict[name] = group.to_dict("records")
    return groups_dict


def get_logs_from_group(group_list):
    logs_from_group = []
    for ele in group_list:
        logs_from_group.append(ele["Content"])
    return logs_from_group


def check_group_count(groups_dict, removed_items=[]):
    for eventID, logs in list(groups_dict.items()):
        if len(logs) < 5:
            removed_items.extend(
                [[log["Content"], log["EventId"], log["EventTemplate"]] for log in logs]
            )
            del groups_dict[eventID]
    print(f"Number of logs in removed small groups: {len(removed_items)}")
    return removed_items, groups_dict


def res_list_to_file(res_list, out_path, regex_sample):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    file_path = os.path.join(out_path, str(regex_sample) + ".csv")
    if not hasattr(res_list_to_file, "first_write"):
        res_list_to_file.first_write = {}
    if file_path not in res_list_to_file.first_write:
        res_list_to_file.first_write[file_path] = True

    def generate_valid_rows():
        for item in res_list:
            try:
                line_id = int(item[0])
                yield (line_id, item[1], item[2], item[3], item[4])
            except (ValueError, IndexError):
                continue

    with open(file_path, "w" if res_list_to_file.first_write[file_path] else "a", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        if res_list_to_file.first_write[file_path]:
            writer.writerow(["LineId", "Content", "EventId", "RegexTemplate", "TemplateHistory"])
            res_list_to_file.first_write[file_path] = False
            
        writer.writerows(generate_valid_rows())

    return file_path

def one_result_to_file(one_result, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    with open(out_path + str(regex_sample) + ".csv", "a", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(one_result)
    return out_path + str(regex_sample) + ".csv"


def reorder_csv_in_place(csv_path, order_list):
    data = []
    with open(csv_path, mode="r", newline="") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        data = list(reader)

    rows_by_key = {row[0]: row for row in data if row}

    sorted_data = []
    for key in order_list:
        if key in rows_by_key:
            sorted_data.append(rows_by_key[key])

    with open(csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(sorted_data)


def prepare_results(output_dir, parser_name, sample_size, list_to_insert, order_list):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    result_file = "summary_[parser={},sample_size={}].csv".format(
        str(parser_name), str(sample_size)
    )
    result_file_path = os.path.join(output_dir, result_file)

    if not os.path.exists(result_file_path) or os.stat(result_file_path).st_size == 0:
        with open(result_file_path, "w", newline="") as csv_file:
            fw = csv.writer(csv_file)
            fw.writerow(
                [
                    "Dataset",
                    "Total_time",
                    "LLaMA_parsing_time",
                    "Drain_parsing_time",
                    "Regex_parsing_time",
                    "RoBERTa_parsing_time",
                    "GA",
                    "PA",
                    "FGA",
                    "FTA",
                    "Event_count",
                    "Total_Tokens",
                    "Average_Tokens"
                ]
            )

    with open(result_file_path, "a", newline="") as csv_file:
        fw = csv.writer(csv_file)
        fw.writerow(list_to_insert)
    reorder_csv_in_place(result_file_path, order_list)
    return result_file


def sort_dict_by_content_length(input_dict):
    def count_words_in_content(entry):
        return len(entry["Content"].split())

    sorted_items = sorted(
        input_dict.items(), key=lambda item: count_words_in_content(item[1][0])
    )

    sorted_dict = {key: value for key, value in sorted_items}
    return sorted_dict


def append_unique_to_csv(data_list, file_path):
    new_data = pd.DataFrame(data_list)
    file = Path(file_path)

    if "Count" in new_data.columns:
        new_data = new_data.drop(columns="Count")
    new_data = new_data.groupby(new_data.columns.tolist(), as_index=False).size()
    new_data = new_data.rename(columns={"size": "Count"})

    if file.is_file():
        existing_data = pd.read_csv(file_path, dtype={1: str})
    else:
        existing_data = pd.DataFrame(columns=new_data.columns)

    combined_data = pd.concat([existing_data, new_data], ignore_index=True)

    combined_data.to_csv(file_path, index=False, header=True)
    return file_path


def align_data_by_lineid(df_gtlog, df_parsedlog):
    """Align ground truth and parsing results by LineId"""
    if 'LineId' not in df_gtlog.columns or 'LineId' not in df_parsedlog.columns:
        print("Warning: LineId column not found in both dataframes")
        return df_gtlog, df_parsedlog

    common_lineids = set(df_gtlog['LineId']).intersection(set(df_parsedlog['LineId']))
    print(f"Found {len(common_lineids)} common LineIds")

    df_gtlog_aligned = df_gtlog[df_gtlog['LineId'].isin(common_lineids)]
    df_parsedlog_aligned = df_parsedlog[df_parsedlog['LineId'].isin(common_lineids)]

    return df_gtlog_aligned, df_parsedlog_aligned

order_list = [
    "HDFS",
    "Hadoop",
    "Spark",
    "Zookeeper",
    "BGL",
    "HPC",
    "Thunderbird",
    "Windows",
    "Linux",
    "Android",
    "HealthApp",
    "Apache",
    "Proxifier",
    "OpenSSH",
    "OpenStack",
    "Mac",
    "TestHadoop",
    "TestLinux",
    "TestOpenStack",
    "Testxieyi",
    "TestBGL",
    "TestSpark",
    "TestMac"
]

if __name__ == "__main__":
    path_prefix = "../result_offline_similar/"
    if "chatglm" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(
            "../models/chatglm3-6b", trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            "../models/chatglm3-6b", trust_remote_code=True, device="cuda"
        )
        model = model.eval()
        pipeline = (model, tokenizer)
    else:
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="cuda:0",
            trust_remote_code=True,
        )
    print("Loading Llama-3 based static marker extractor...", flush=True)
    #marker_extractor = LLMMarkerExtractor(pipeline_instance=pipeline)
    print("LLM extractor loaded.", flush=True)
    #print("Loading unsupervised marker extraction model (RoBERTa)...", flush=True)
    marker_extractor = UnsupervisedMarkerExtractor(model_path="../models/roberta-base")
    #print("RoBERTa model loaded.", flush=True)

    print(f"{model_path} Pipeline is ready.", flush=True)
    for system in datasets_full:
        print(f"Start Parsing {system}", flush=True)
        marker_extractor.total_time = 0.0
        if system == "Linux":
            log_file = f"../full_dataset/{system}/{system}_full.log_structured.csv"
        else:
            log_file = f"../full_dataset/{system}/{system}_full.log_structured.csv"
        out_path = f"{path_prefix}{system}/"
        if not os.path.exists(path_prefix):
            os.makedirs(path_prefix)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        setting = benchmark_settings[system]
        start_time = datetime.now()
        Drain_parser1 = grouping.LogParser(
            rex=setting["regex"], depth=setting["depth"], st=setting["st"]
        )
        logs = read_column_from_csv(log_file)
        grouped_logs = Drain_parser1.parse(logs)
        groups_dict = group_logs_using_parser(grouped_logs)
        groups_dict = sort_dict_by_content_length(groups_dict)
        print("==================", flush=True)
        print(
            "initial set grouping finished, start parsing. ",
            len(groups_dict.keys()),
            " groups in total for ",
            len(logs),
            " logs",
            flush=True,
        )
        print("==================", flush=True)
        regex_manager1 = regex_manager.RegexTemplateManager()
        llama_parser1 = llama_parser.LogParser(
            pipeline=pipeline,
            model=model_path,
            regex_manager1=regex_manager1,
            regex_sample=regex_sample,
            similarity=similarity,
            do_self_reflection=do_self_reflection,
            markers=system,  
            marker_extractor=marker_extractor  
        )
        for eventid in tqdm(groups_dict.keys(), desc=f"Processing events {system}"):
            append_unique_to_csv(groups_dict[eventid], out_path + "group.csv")
            res_list = []
            logs_from_group = [{"Content": log["Content"], "LineId": log["LineId"], "EventId": eventid} for log in groups_dict[eventid]]
            res_list = llama_parser1.parse(groups_dict[eventid], logs_from_group)
            print(f"Predicted result list (res_list) lines: {len(res_list)}")
            out_file = res_list_to_file(res_list, out_path, regex_sample=regex_sample)
            gc.collect()
            torch.cuda.empty_cache()
        Drain_parser1.print_time()
        regex_manager1.print_time()
        regex_manager1.print_regex_templates()
        total_time = datetime.now() - start_time
        print(
            system + " Parsing done. [Time taken: {!s}]".format(total_time), flush=True
        )
        if system == "Linux":
            ground_truth_file = f"../full_dataset/{system}/{system}_full.log_structured.csv"
        else:
            ground_truth_file = f"../full_dataset/{system}/{system}_full.log_structured.csv"
        file_path = f"{out_path}/{str(regex_sample)}.csv"
        sorted_file = f"{out_path}/{str(regex_sample)}_sorted.csv"

        file_path = f"{out_path}/{str(regex_sample)}.csv"  
        sort_csv_by_lineid(file_path)  

        GA, PA, FGA, FTA, event_count = accuracy.evaluate_result(
            file_path, ground_truth_file,
        )
        llama_pure_time = llama_parser1.total_time - regex_manager1.total_time - marker_extractor.total_time
        total_tokens = llama_parser1.total_tokens
        total_calls = llama_parser1.total_calls
        avg_tokens = total_tokens / total_calls if total_calls > 0 else 0.0
        
        print("==================", flush=True)
        print(f"[{system}] Token Stats - Total: {total_tokens}, Avg: {avg_tokens:.1f}, Calls: {total_calls}", flush=True)
        print(
            system,
            total_time,
            llama_pure_time,
            Drain_parser1.total_time,
            regex_manager1.total_time,
            marker_extractor.total_time,
            GA,
            PA,
            FGA,
            FTA,
            event_count,
            total_tokens,       
            round(avg_tokens,1),
            flush=True,
        )
        prepare_results(
            output_dir=path_prefix,
            parser_name="Drain",
            sample_size=regex_sample,
            list_to_insert=[
                system,
                total_time.total_seconds(),
                llama_pure_time,
                Drain_parser1.total_time,
                regex_manager1.total_time,
                marker_extractor.total_time,
                GA,
                PA,
                FGA,
                FTA,
                event_count,
                total_tokens,       
                round(avg_tokens,1) 
            ],
            order_list=order_list,
        )
        print("==================", flush=True)
        evaluate_roberta_markers(
            system=system, 
            records=marker_extractor.scoring_records, 
            ground_truth_file=ground_truth_file,
            out_csv="roberta_marker_stats.csv" 
        )
        marker_extractor.scoring_records.clear()
