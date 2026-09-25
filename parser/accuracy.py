import pandas as pd
import os
import numpy as np

def evaluate_result(predic_file,groundtruth):
    df_gtlog = pd.read_csv(
        groundtruth,  usecols=["Content", "EventId", "EventTemplate", "LineId"]
    )
    print("df_gtlog file loaded! ", flush=True)
    df_gtlog["EventTemplate_NoSpaces"] = df_gtlog["EventTemplate"].str.replace('\s+', '', regex=True).str.replace(r"'<\*>'", "<*>", regex=True).str.replace(r'\<\*\>', '', regex=True).str.replace(',', '').str.replace(':', '').str.replace('\$', '', regex=True).str.replace(r'\+', '', regex=True).str.replace(r'\[\s*\]', '', regex=True).str.replace(r'\(\)', '', regex=True)
    print("df_gtlog EventTemplate ready to be checked", flush=True)
    column_names = ['Content',  'RegexTemplate', 'LineId', 'TemplateHistory']
    df_parsedlog = pd.read_csv(predic_file, index_col=False,  usecols=column_names)
    print("df_parsedlog file loaded! ", flush=True)
    print(f"Number of predicted lines read: {len(df_parsedlog)}")
    df_parsedlog["Predict_NoSpaces"] = df_parsedlog['RegexTemplate'].str.replace('\s+', '', regex=True).str.replace(r'\(\.\*\?\)', '', regex=True).str.replace(r'\\', '', regex=True).str.replace(',', '').str.replace(':', '').str.replace('\$', '', regex=True).str.replace(r'\+', '', regex=True).str.replace(r'\[\s*\]', '', regex=True).str.replace(r'\(\)', '', regex=True)
    print("df_parsedlog ready to be checked! ", flush=True)
    # df_parsedlog["EventTemplate_NoSpaces"] = df_parsedlog['EventTemplate'].str.replace('\s+', '', regex=True)

    if len(df_parsedlog) != len(df_gtlog):
        raise ValueError(f"Number of predicted lines ({len(df_parsedlog)}) does not match ground truth lines ({len(df_gtlog)})!")


    assert len(df_parsedlog) == len(df_gtlog), f"Number of predicted log lines ({len(df_parsedlog)}) does not match ground truth log lines ({len(df_gtlog)})!"
    content_match = df_parsedlog['Content'].eq(df_gtlog['Content']).all()
    assert content_match, "The Content field of predicted logs does not match the ground truth. There might be order errors or content modifications!"

    unmatched_mask = df_parsedlog["Predict_NoSpaces"] != df_gtlog["EventTemplate_NoSpaces"]
    target_indices = df_gtlog.loc[unmatched_mask, 'EventTemplate'].drop_duplicates().index

    df_error_analysis = pd.DataFrame({
        'LineId': df_gtlog.loc[target_indices, 'LineId'],
        'GroundTruth_Original': df_gtlog.loc[target_indices, 'EventTemplate'],
        'GroundTruth_Processed': df_gtlog.loc[target_indices, 'EventTemplate_NoSpaces'],
        'Predicted_Original': df_parsedlog.loc[target_indices, 'RegexTemplate'],
        'Thinking_Process': df_parsedlog.loc[target_indices, 'TemplateHistory'].fillna(""),
        'Predicted_Processed': df_parsedlog.loc[target_indices, 'Predict_NoSpaces']
    })
    if not df_error_analysis.empty:
        dir_name = os.path.dirname(predic_file)
        base_name = os.path.basename(predic_file).replace('.csv', '')
        error_output_path = os.path.join(dir_name, f"{base_name}_error_analysis.csv")

        df_error_analysis.to_csv(error_output_path, index=False, encoding='utf-8-sig')
        print(f"❌ Found {len(df_error_analysis)} unmatched logs.", flush=True)
        print(f"📂 Detailed error analysis report generated: {error_output_path}", flush=True)
    else:
        print("✅ Perfect match! No errors found, no error report generated.", flush=True)

    correctly_parsed_messages = df_parsedlog['Predict_NoSpaces'].eq(df_gtlog['EventTemplate_NoSpaces']).values.sum()
    PA = float(correctly_parsed_messages) / len(df_parsedlog[['Content']])
    print(f"PA: {PA}", flush=True)
    # print(f"PA: {PA}", flush=True)
    GA, FGA, FTA = get_accuracy(df_gtlog["EventTemplate_NoSpaces"], df_parsedlog['Predict_NoSpaces'])

    print(f"accuracy_GA: {GA}", flush=True)
    print(f"accuracy_FGA: {FGA}", flush=True)
    print(f"accuracy_FTA: {FTA}", flush=True)
    event_count = str(df_parsedlog["Predict_NoSpaces"].nunique())

    return GA, PA, FGA, FTA, event_count


def get_accuracy(series_groundtruth, series_parsedlog, debug=False):
    series_parsedlog_valuecounts = series_parsedlog.value_counts()
    series_groundtruth_valuecounts = series_groundtruth.value_counts()

    accurate_events = 0
    accurate_groups = 0
    accurate_templates = 0

    for parsed_eventId in series_parsedlog_valuecounts.index:
        logIds = series_parsedlog[series_parsedlog == parsed_eventId].index
        series_groundtruth_logId_valuecounts = series_groundtruth[logIds].value_counts()
        error_eventIds = (
            parsed_eventId,
            series_groundtruth_logId_valuecounts.index.tolist(),
        )
        error = True
        if series_groundtruth_logId_valuecounts.size == 1:
            groundtruth_eventId = series_groundtruth_logId_valuecounts.index[0]
            if (
                    logIds.size
                    == series_groundtruth[series_groundtruth == groundtruth_eventId].size
            ):
                accurate_events += logIds.size
                accurate_groups += 1
                if parsed_eventId == groundtruth_eventId:
                    accurate_templates += 1
                error = False
        if error and debug:
            print(
                "(parsed_eventId, groundtruth_eventId) =",
                error_eventIds,
                "failed",
                logIds.size,
                "messages",
            )

    GA = float(accurate_events) / series_groundtruth.size

    total_predicted_groups = len(series_parsedlog_valuecounts)
    total_groundtruth_groups = len(series_groundtruth_valuecounts)

    FGA_precision = accurate_groups / total_predicted_groups if total_predicted_groups > 0 else 0
    FGA_recall = accurate_groups / total_groundtruth_groups if total_groundtruth_groups > 0 else 0
    FGA = 2 * (FGA_precision * FGA_recall) / (FGA_precision + FGA_recall) if (FGA_precision + FGA_recall) > 0 else 0

    FTA_precision = accurate_templates / total_predicted_groups if total_predicted_groups > 0 else 0
    FTA_recall = accurate_templates / total_groundtruth_groups if total_groundtruth_groups > 0 else 0
    FTA = 2 * (FTA_precision * FTA_recall) / (FTA_precision + FTA_recall) if (FTA_precision + FTA_recall) > 0 else 0

    return GA, FGA, FTA
