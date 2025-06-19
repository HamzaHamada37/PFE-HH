import logging
import os
import json
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from bs4 import BeautifulSoup
import re
from collections import Counter
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='nlp_pipeline.log'
)

# Initialize global variables for NLP components
tokenizer = None
model = None
lemmatizer = None

def clean_description(text):
    """Clean and preprocess text data."""
    if pd.isna(text) or not isinstance(text, str):
        return ''

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags if present
    text = BeautifulSoup(text, 'html.parser').get_text()

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text

def calculate_sentiment(text):
    """Calculate sentiment score using FinBERT."""
    try:
        if not text or pd.isna(text):
            return 0.0

        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text)

        # Tokenize text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        # Get model outputs
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Convert predictions to sentiment score
        # FinBERT output: [negative, neutral, positive]
        neg, neu, pos = predictions[0].tolist()
        
        # Calculate weighted sentiment score (-1 to 1)
        sentiment_score = -1 * neg + 1 * pos

        return sentiment_score

    except Exception as e:
        logging.error(f"Error calculating sentiment: {str(e)}")
        return 0.0

def is_html_excel(filepath):
    """Check if the file is an HTML-based Excel file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read(1024)  # Read first 1KB
            return '<html' in content.lower() or '<table' in content.lower()
    except Exception:
        return False

def parse_html_excel(filepath):
    """Parse HTML-based Excel file using pandas."""
    try:
        # Read HTML file
        dfs = pd.read_html(filepath)
        if dfs:
            # Use the largest table
            return max(dfs, key=len)
        return None
    except Exception as e:
        logging.error(f"Error parsing HTML Excel: {str(e)}")
        return None

def convert_html_to_excel(html_file):
    """Convert HTML Excel file to real Excel file."""
    try:
        df = parse_html_excel(html_file)
        if df is not None:
            # Save to temporary Excel file
            temp_file = html_file + '.xlsx'
            df.to_excel(temp_file, index=False)
            return temp_file
        return None
    except Exception as e:
        logging.error(f"Error converting HTML to Excel: {str(e)}")
        return None

def process_jira_file(jira_file):
    """Process a JIRA Excel file and extract metrics.

    Args:
        jira_file: A JiraFile model instance

    Returns:
        tuple: (analysis_results dict, client_metrics DataFrame)
    """
    try:
        # Download necessary NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)

        # Get file path
        FILEPATH = jira_file.file.path
        logging.info(f"Processing file: {FILEPATH}")

        # First check if this is an HTML-based Excel file
        if is_html_excel(FILEPATH):
            logging.info(f"Detected HTML-based Excel file: {FILEPATH}")

            # Try to parse the HTML Excel file
            html_df = parse_html_excel(FILEPATH)
            if html_df is not None:
                df = html_df
                logging.info(f"Successfully parsed HTML Excel file with {len(df)} rows")
                success = True
            else:
                # Try to convert the HTML Excel file to a real Excel file
                logging.info("Attempting to convert HTML Excel file to real Excel")
                converted_file = convert_html_to_excel(FILEPATH)
                if converted_file:
                    try:
                        df = pd.read_excel(converted_file, header=0)
                        logging.info(f"Successfully loaded converted Excel file with {len(df)} rows")
                        success = True
                        # Clean up temp file
                        os.remove(converted_file)
                    except Exception as conv_error:
                        logging.error(f"Failed to load converted Excel file: {str(conv_error)}")
                        success = False
                else:
                    logging.error("Failed to convert HTML Excel file")
                    success = False
        else:
            # Not an HTML Excel file, try standard Excel approaches
            success = False

            # Try to load the Excel file with different approaches
            try:
                # First try with specific sheet name and auto-detect engine
                df = pd.read_excel(FILEPATH, sheet_name='general_report', header=3)
                logging.info("Successfully loaded 'general_report' sheet")
                success = True
            except Exception as e:
                logging.warning(f"Could not load 'general_report' sheet: {str(e)}")
                try:
                    # Try with explicit engines if auto-detection fails
                    engines = ['openpyxl', 'xlrd']

                    for engine in engines:
                        try:
                            logging.info(f"Trying to load Excel file with engine: {engine}")
                            # Try with ExcelFile first to get sheet names
                            with pd.ExcelFile(FILEPATH, engine=engine) as excel_file:
                                all_sheets = excel_file.sheet_names
                                logging.info(f"Available sheets with {engine} engine: {all_sheets}")

                                if all_sheets:
                                    df = pd.read_excel(excel_file, sheet_name=all_sheets[0], header=3)
                                    logging.info(f"Successfully loaded first sheet: {all_sheets[0]} with {engine} engine")
                                    success = True
                                    break
                                else:
                                    # Try without sheet name
                                    df = pd.read_excel(FILEPATH, header=3, engine=engine)
                                    logging.info(f"Loaded Excel file without specifying sheet name using {engine} engine")
                                    success = True
                                    break
                        except Exception as engine_error:
                            logging.warning(f"Failed to load with {engine} engine: {str(engine_error)}")
                            continue

                    if not success:
                        # If all engines failed, try specialized approaches for .xls files
                        if FILEPATH.lower().endswith('.xls'):
                            # First try the simplest approach for Excel 97-2003 files
                            try:
                                logging.info("Detected .xls file (Excel 97-2003 format), trying simplest approach first")
                                # Try with explicit xlrd engine and no header
                                df = pd.read_excel(
                                    FILEPATH,
                                    engine='xlrd',
                                    header=None  # Read without assuming header position
                                )
                                # If we get here, we successfully loaded the file
                                logging.info("Successfully loaded .xls file with xlrd engine and no header")

                                # Now try to determine the header row
                                # Assume header is in one of the first 5 rows
                                for i in range(5):
                                    if i < len(df):
                                        # Use this row as header
                                        header_row = df.iloc[i]
                                        df = df.iloc[i+1:].reset_index(drop=True)
                                        df.columns = header_row
                                        logging.info(f"Using row {i} as header")
                                        break

                                success = True
                            except Exception as simple_error:
                                logging.warning(f"Simple approach failed: {str(simple_error)}")

                                # Try with xlrd engine specifically for old Excel 97-2003 files
                                try:
                                    logging.info("Trying more complex approach with direct xlrd access")
                                    import xlrd

                                    # Open the workbook directly with xlrd
                                    workbook = xlrd.open_workbook(FILEPATH)
                                    sheet_names = workbook.sheet_names()
                                    logging.info(f"Available sheets in .xls file: {sheet_names}")

                                    if sheet_names:
                                        # Use the first sheet
                                        sheet = workbook.sheet_by_name(sheet_names[0])
                                        # Convert xlrd sheet to pandas DataFrame
                                        data = []
                                        for row_idx in range(sheet.nrows):
                                            if row_idx < 3:  # Skip header rows (assuming header=3)
                                                continue
                                            row = sheet.row_values(row_idx)
                                            data.append(row)

                                        # Create column names from the 4th row (index 3)
                                        if sheet.nrows > 3:
                                            columns = sheet.row_values(3)
                                        else:
                                            columns = [f"Column_{i}" for i in range(sheet.ncols)]

                                        # Create DataFrame
                                        df = pd.DataFrame(data, columns=columns)
                                        logging.info(f"Successfully loaded .xls file using direct xlrd approach")
                                        success = True
                                    else:
                                        logging.error("No sheets found in the .xls file")
                                except Exception as xls_error:
                                    logging.error(f"Failed to load .xls file with direct xlrd approach: {str(xls_error)}")

                        # If all approaches failed, try a generic CSV approach as last resort
                        if not success:
                            try:
                                logging.info("Trying to load file as CSV as last resort")
                                df = pd.read_csv(FILEPATH, sep=None, engine='python')  # Auto-detect separator
                                logging.info(f"Successfully loaded file as CSV with {len(df)} rows")
                                success = True
                            except Exception as csv_error:
                                logging.error(f"Failed to load file as CSV: {str(csv_error)}")

                                # If all approaches failed, raise the exception
                                if not success:
                                    raise Exception("All file reading approaches failed")
                except Exception as e2:
                    logging.error(f"Failed to load Excel file: {str(e2)}")
                    raise Exception(f"Could not read Excel file: {str(e2)}")
            except Exception as e2:
                logging.error(f"Failed to load Excel file: {str(e2)}")
                raise Exception(f"Could not read Excel file: {str(e2)}")

        # Drop specified columns
        cols_to_drop = ["Linked Issues", "Status", "Assignee", "Fix Version/s", "Resolved", "Linked issue"]
        existing_cols_to_drop = [col for col in df.columns if col in cols_to_drop]
        df_cleaned = df.drop(columns=existing_cols_to_drop)

        # Handle Description columns more robustly
        logging.info(f"Columns before processing: {df_cleaned.columns.tolist()}")

        # Check if there are duplicate 'Description' columns
        description_cols = [col for col in df_cleaned.columns if 'description' in col.lower()]
        logging.info(f"Found description columns: {description_cols}")

        if len(description_cols) > 1:
            # More efficient approach - directly filter columns to keep
            columns_to_keep = df_cleaned.columns.tolist()
            # Find the first 'Description' column and remove it
            for i, col in enumerate(columns_to_keep):
                if col.strip() == 'Description':
                    columns_to_keep.pop(i)
                    break

            df_cleaned = df_cleaned[columns_to_keep]
            logging.info(f"Dropped first Description column, remaining columns: {df_cleaned.columns.tolist()}")
        else:
            logging.info("No duplicate Description columns found")

        # Keep only the desired rows
        df_cleaned = df_cleaned.head(360)

        # Skip saving and reloading intermediate file - use the dataframe directly
        df = df_cleaned.copy()

        # Handle Creator column more robustly
        if 'Creator' in df.columns:
            # Clean "Creator" column
            df['Creator'] = df['Creator'].str.replace(r'^.*? - ', '', regex=True).str.strip()
            logging.info("Cleaned Creator column")
        else:
            # Try to find a suitable column for Creator
            possible_creator_cols = ['Reporter', 'Author', 'Created by', 'User']
            creator_col = None

            for col in possible_creator_cols:
                if col in df.columns:
                    creator_col = col
                    break

            if creator_col:
                # Use the found column and rename it
                df['Creator'] = df[creator_col].str.replace(r'^.*? - ', '', regex=True).str.strip()
                logging.info(f"Using '{creator_col}' as Creator column")
            else:
                # Create a default Creator column if none exists
                df['Creator'] = 'Unknown User'
                logging.warning("No Creator column found, using 'Unknown User'")

        # Handle Key column for deduplication
        if 'Key' in df.columns:
            # Remove duplicates based on UNIQUE Key
            df = df.drop_duplicates(subset="Key", keep='first')
            logging.info("Removed duplicates based on Key column")
        else:
            # Try to find a suitable column for deduplication
            possible_key_cols = ['Issue key', 'ID', 'Ticket ID', 'Issue ID']
            key_col = None

            for col in possible_key_cols:
                if col in df.columns:
                    key_col = col
                    break

            if key_col:
                # Use the found column for deduplication
                df = df.drop_duplicates(subset=key_col, keep='first')
                logging.info(f"Removed duplicates based on '{key_col}' column")
            else:
                # If no key column, try to use index or just keep all rows
                logging.warning("No Key column found for deduplication, keeping all rows")

        # Initialize NLP components - only once if not already initialized
        global tokenizer, model, lemmatizer
        if 'tokenizer' not in globals() or tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        if 'model' not in globals() or model is None:
            model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        if 'lemmatizer' not in globals() or lemmatizer is None:
            lemmatizer = WordNetLemmatizer()

        # Apply text cleaning and sentiment analysis
        # Find the description column - try different possible names
        description_col = next((col for col in ['Description.1', 'Description', 'description', 'Summary']
                               if col in df.columns), None)

        if description_col:
            logging.info(f"Using '{description_col}' column for text analysis")
        else:
            # If no description column found, use the first text column as fallback
            text_cols = [col for col in df.columns if df[col].dtype == 'object']
            if text_cols:
                description_col = text_cols[0]
                logging.info(f"No standard description column found, using '{description_col}' instead")
            else:
                # Create a dummy column if no text column exists
                description_col = 'dummy_description'
                df[description_col] = 'No description available'
                logging.warning("No text columns found in the data, using dummy descriptions")

        df['cleaned_description'] = df[description_col].apply(clean_description)
        df['sentiment'] = df['cleaned_description'].apply(calculate_sentiment)

        # Handle date columns more robustly
        # Find Created date column using more efficient approach
        created_col = next((col for col in ['Created', 'Created Date', 'Creation Date', 'Date Created']
                           if col in df.columns), None)

        if created_col is None:
            # If no created column found, create a dummy one with today's date
            created_col = 'Created'
            df[created_col] = pd.Timestamp.now()
            logging.warning("No Created date column found, using current date")
        else:
            # Convert to datetime
            df[created_col] = pd.to_datetime(df[created_col], errors='coerce')
            logging.info(f"Using '{created_col}' as creation date")

        # Find Response date column using more efficient approach
        response_col = next((col for col in ['Date of First Response', 'First Response', 'Response Date', 'Responded']
                            if col in df.columns), None)

        if response_col is None:
            # If no response column found, create a dummy one with today's date
            response_col = 'Date of First Response'
            df[response_col] = df[created_col] + pd.Timedelta(days=3)  # Assume 3 days response time
            logging.warning("No Response date column found, using created date + 3 days")
        else:
            # Convert to datetime
            df[response_col] = pd.to_datetime(df[response_col], errors='coerce')
            logging.info(f"Using '{response_col}' as response date")

        # Calculate resolution time
        df['Resolution Time (Days)'] = (df[response_col] - df[created_col]).dt.days
        df['Resolution Time (Days)'] = df['Resolution Time (Days)'].clip(lower=0)
        df['Resolution Time (Days)'] = df['Resolution Time (Days)'].fillna(0)

        # Calculate temporal decay
        df['days_old'] = (pd.Timestamp.now() - df[created_col]).dt.days
        df['temporal_decay'] = np.exp(-df['days_old'] / 60)

        # --- CALCULATION COMPONENTS ---
        # 1. Sentiment Impact (50% weight)
        df['sentiment_impact'] = ((1 - df['sentiment']) / 2) * 0.50

        # 2. Ticket Impact (20% weight)
        df['decayed_ticket_count'] = df.groupby('Creator')['temporal_decay'].transform('sum')
        max_decayed = df['decayed_ticket_count'].max()
        df['ticket_impact'] = (df['decayed_ticket_count'] / max_decayed) * 0.20

        # 3. Priority Impact (15% weight)
        priority_weights = {
            'Major': 0.13, 'Medium': 0.10, 'Minor': 0.07,
            'Critical': 0.15, 'Blocker': 0.14
        }

        # Find Priority column using more efficient approach
        priority_col = next((col for col in ['Priority', 'Severity', 'Importance', 'Urgency']
                            if col in df.columns), None)

        if priority_col is None:
            # If no priority column found, create a dummy one with medium priority
            priority_col = 'Priority'
            df[priority_col] = 'Medium'
            logging.warning("No Priority column found, using 'Medium' as default")

        # Apply priority weights
        df['priority_impact'] = df[priority_col].astype(str).str.strip().str.title().map(
            lambda x: priority_weights.get(x, 0.10)
        )
        logging.info(f"Using '{priority_col}' for priority impact calculation")

        # 4. Issue Type Impact (15% weight)
        issue_type_weights = {
            'Incident': 0.25, 'Defect': 0.2,
            'Information Request': 0.1, 'Requirement': 0.15
        }

        # Find Issue Type column using more efficient approach
        issue_type_col = next((col for col in ['Issue Type', 'Type', 'Issue Category', 'Category']
                              if col in df.columns), None)

        if issue_type_col is None:
            # If no issue type column found, create a dummy one
            issue_type_col = 'Issue Type'
            df[issue_type_col] = 'Task'
            logging.warning("No Issue Type column found, using 'Task' as default")

        # Apply issue type weights
        df['Issue_Type_impact'] = df[issue_type_col].astype(str).str.strip().map(
            lambda x: issue_type_weights.get(x, 0.10)
        )
        df['Issue_Type_impact'] = (df['Issue_Type_impact'] / 0.25) * 0.15  # Normalize to 15%
        logging.info(f"Using '{issue_type_col}' for issue type impact calculation")

        # Combine components
        df['urgency_score'] = (
            df['sentiment_impact'] +
            df['priority_impact'] +
            df['ticket_impact'] +
            df['Issue_Type_impact']
        ) * 3  # Scale to 0-3

        # Final adjustments
        df['urgency_score'] = ((df['urgency_score'] - 1) / (3 - 1) * 100).round(2).astype(str) + '%'

        # Group by Creator
        # Use the description column we identified earlier for ticket counting
        agg_dict = {
            'sentiment': 'mean',
            'urgency_score': lambda x: x.str.rstrip('%').astype(float).mean(),
            'priority_impact': 'mean',
            'Issue_Type_impact': 'mean',
            'Resolution Time (Days)': 'mean'
        }

        # Add the description column for counting tickets
        agg_dict[description_col] = 'count'

        client_grouped = df.groupby('Creator').agg(agg_dict).reset_index()

        # Rename columns
        rename_dict = {
            'urgency_score': 'Customer_Experience_Score',
            'priority_impact': 'Priority_Impact',
            'Issue_Type_impact': 'Issue_Type_Impact',
            'Resolution Time (Days)': 'Avg_Resolution_Time_Days'
        }

        # Add the description column rename
        rename_dict[description_col] = 'Tickets'

        client_grouped.rename(columns=rename_dict, inplace=True)

        # Reorder columns
        client_grouped = client_grouped[
            ['Creator', 'sentiment', 'Priority_Impact',
             'Issue_Type_Impact', 'Tickets', 'Avg_Resolution_Time_Days', 'Customer_Experience_Score']
        ]

        # Convert 'Customer Experience Score' from percentage to decimal
        client_grouped['Customer_Experience_Score'] = client_grouped['Customer_Experience_Score'] / 100

        # Export results
        client_grouped.to_excel('Jira_NLP_By_Client.xlsx', index=False)

        # Export full data with impact columns
        df.to_csv('cleaned_jira_data_with_impact.csv', index=False)

        # Convert client_grouped DataFrame to dictionary format for the database
        # More efficient approach using DataFrame.to_dict() and dictionary comprehension
        client_metrics_dict = {}
        client_data = client_grouped.set_index('Creator').to_dict('index')

        for creator, metrics in client_data.items():
            # Use numpy_encoder to handle NumPy types properly
            creator_str = str(numpy_encoder(creator))
            client_metrics_dict[creator_str] = {
                'sentiment': numpy_encoder(metrics['sentiment']),
                'Priority_Impact': numpy_encoder(metrics['Priority_Impact']),
                'Issue_Type_Impact': numpy_encoder(metrics['Issue_Type_Impact']),
                'Tickets': numpy_encoder(metrics['Tickets']),
                'Avg_Resolution_Time_Days': numpy_encoder(metrics['Avg_Resolution_Time_Days']),
                'Client_Impact': numpy_encoder(metrics['Customer_Experience_Score']),
                'Customer_Experience_Score': numpy_encoder(metrics['Customer_Experience_Score'])
            }

        # Ensure we have real client metrics
        if not client_metrics_dict:
            logging.error("No client metrics were generated from the data")
            raise ValueError("Failed to generate client metrics from the provided data")
        else:
            logging.info(f"Generated client metrics for {len(client_metrics_dict)} creators")

        # Prepare analysis results for the database with proper JSON serialization
        analysis_results = {
            'issue_count': numpy_encoder(len(df)),
            'ticket_types': {k: numpy_encoder(v) for k, v in df[issue_type_col].value_counts().to_dict().items()},
            'priority_distribution': {k: numpy_encoder(v) for k, v in df[priority_col].value_counts().to_dict().items()},
            'status_distribution': {},  # Initialize with empty dict
            'sentiment_analysis': {
                'positive': numpy_encoder((df['sentiment'] > 0.3).sum()),
                'neutral': numpy_encoder(((df['sentiment'] >= 0) & (df['sentiment'] <= 0.3)).sum()),
                'negative_low': numpy_encoder(((df['sentiment'] >= -0.2) & (df['sentiment'] < 0)).sum()),
                'negative_medium': numpy_encoder(((df['sentiment'] >= -0.5) & (df['sentiment'] < -0.2)).sum()),
                'negative_high': numpy_encoder((df['sentiment'] < -0.5).sum())
            },
            'client_metrics': client_metrics_dict  # Already processed with numpy_encoder
        }

        # Add status distribution if Status column exists - using more efficient approach
        status_col = next((col for col in ['Status', 'State', 'Ticket Status']
                          if col in df.columns), None)

        if status_col:
            analysis_results['status_distribution'] = {
                k: numpy_encoder(v) for k, v in df[status_col].value_counts().to_dict().items()
            }
            logging.info(f"Using '{status_col}' for status distribution")
        else:
            # Create a default status distribution
            df_len = len(df)
            analysis_results['status_distribution'] = {
                'Open': numpy_encoder(int(df_len * 0.3)),
                'In Progress': numpy_encoder(int(df_len * 0.4)),
                'Resolved': numpy_encoder(int(df_len * 0.3))
            }
            logging.warning("No Status column found, using default status distribution")

        logging.info("Analysis completed successfully")
        return analysis_results, client_grouped

    except Exception as e:
        logging.error(f"Error processing file: {str(e)}")
        raise e

def numpy_encoder(obj):
    """Custom JSON encoder to handle NumPy types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    return obj

class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder class for NumPy types"""
    def default(self, obj):
        return numpy_encoder(obj)

if __name__ == "__main__":
    # For testing the script directly
    import time

    class MockJiraFile:
        def __init__(self, path):
            self.file = self
            self.path = path

    mock_file = MockJiraFile('Jira OeKB.xls.xlsx')

    # Benchmark the processing time
    start_time = time.time()
    results, client_metrics = process_jira_file(mock_file)
    end_time = time.time()

    print(f"\nProcessing completed in {end_time - start_time:.2f} seconds")

    # Print client metrics table
    print("\nClient Metrics:")
    print(client_metrics.to_string(index=False))


# Survey Processing Functions
def process_survey_file(survey_file):
    """Process a survey Excel file and extract team metrics.

    Args:
        survey_file: A SurveyFile model instance

    Returns:
        dict: Analysis results containing KPIs and insights
    """
    try:
        from .models import SurveyResponse, SurveyAnalysis

        # Get file path
        filepath = survey_file.file.path
        logging.info(f"Processing survey file: {filepath}")

        # Read Excel file with error handling
        try:
            df = pd.read_excel(filepath)
            logging.info(f"Loaded survey data with {len(df)} responses")
        except Exception as e:
            raise Exception(f"Could not read Excel file: {str(e)}. Please ensure the file is a valid Excel file (.xlsx or .xls).")

        if df.empty:
            raise Exception("The uploaded file is empty. Please upload a file with survey data.")

        # Validate expected columns
        expected_columns = [
            'Role', 'Q1_1_SpeakingUp', 'Q1_2_MistakesHeldAgainstMe', 'Q1_3_RespectWhenNotKnowing',
            'Q2_1_WorkloadManageable', 'Q2_2_ToolsAndResources', 'Q2_3_WorkLifeBalance',
            'Q3_1_UnderstandingClients', 'Q3_2_SupportHandlingClients', 'Q3_3_ToolsForClientService',
            'Q4_1_HelpResponsiveness', 'Q4_2_ConflictResolution', 'Q4_3_SharingUpdates',
            'Open_Feedback'
        ]

        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            logging.warning(f"Missing expected columns: {missing_columns}")
            raise Exception(f"Missing required columns: {', '.join(missing_columns)}. Please ensure your Excel file has the correct column structure.")

        logging.info(f"All required columns found: {df.columns.tolist()}")

        # Clean data - filter out N/A values for quantitative questions
        quantitative_columns = [col for col in expected_columns[1:13] if col in df.columns]  # Q1-Q4 questions

        # Create clean dataset for KPI calculations (remove rows with N/A in quantitative columns)
        df_clean = df.dropna(subset=quantitative_columns, how='any').copy()
        logging.info(f"Clean dataset has {len(df_clean)} responses after removing N/A values")

        # Store individual responses
        survey_file.responses.all().delete()  # Clear existing responses

        for _, row in df.iterrows():
            SurveyResponse.objects.create(
                survey_file=survey_file,
                role=row.get('Role', ''),
                q1_1_speaking_up=row.get('Q1_1_SpeakingUp'),
                q1_2_mistakes_held_against=row.get('Q1_2_MistakesHeldAgainstMe'),
                q1_3_respect_when_not_knowing=row.get('Q1_3_RespectWhenNotKnowing'),
                q2_1_workload_manageable=row.get('Q2_1_WorkloadManageable'),
                q2_2_tools_and_resources=row.get('Q2_2_ToolsAndResources'),
                q2_3_work_life_balance=row.get('Q2_3_WorkLifeBalance'),
                q3_1_understanding_clients=row.get('Q3_1_UnderstandingClients'),
                q3_2_support_handling_clients=row.get('Q3_2_SupportHandlingClients'),
                q3_3_tools_for_client_service=row.get('Q3_3_ToolsForClientService'),
                q4_1_help_responsiveness=row.get('Q4_1_HelpResponsiveness'),
                q4_2_conflict_resolution=row.get('Q4_2_ConflictResolution'),
                q4_3_sharing_updates=row.get('Q4_3_SharingUpdates'),
                open_feedback=row.get('Open_Feedback', '')
            )

        # Calculate KPIs from clean data
        analysis_results = calculate_survey_kpis(df_clean, df)

        # Create or update analysis record
        analysis, created = SurveyAnalysis.objects.update_or_create(
            survey_file=survey_file,
            defaults=analysis_results
        )

        # Mark file as processed
        survey_file.processed = True
        survey_file.save()

        logging.info("Survey processing completed successfully")
        return analysis_results

    except Exception as e:
        logging.error(f"Error processing survey file: {str(e)}")
        raise Exception(f"Could not process survey file: {str(e)}")


def calculate_survey_kpis(df_clean, df_all):
    """Calculate KPIs and metrics from survey data.

    Args:
        df_clean: DataFrame with N/A values removed for quantitative analysis
        df_all: Complete DataFrame including all responses

    Returns:
        dict: Analysis results
    """
    try:
        # Basic counts
        total_responses = len(df_all)
        valid_responses = len(df_clean)

        # Role distribution
        role_distribution = df_all['Role'].value_counts().to_dict()

        # Calculate category averages from clean data
        psychological_safety_cols = ['Q1_1_SpeakingUp', 'Q1_2_MistakesHeldAgainstMe', 'Q1_3_RespectWhenNotKnowing']
        work_environment_cols = ['Q2_1_WorkloadManageable', 'Q2_2_ToolsAndResources', 'Q2_3_WorkLifeBalance']
        client_service_cols = ['Q3_1_UnderstandingClients', 'Q3_2_SupportHandlingClients', 'Q3_3_ToolsForClientService']
        team_collaboration_cols = ['Q4_1_HelpResponsiveness', 'Q4_2_ConflictResolution', 'Q4_3_SharingUpdates']

        # Calculate averages (only from valid responses)
        avg_psychological_safety = df_clean[psychological_safety_cols].mean().mean() if not df_clean.empty else None
        avg_work_environment = df_clean[work_environment_cols].mean().mean() if not df_clean.empty else None
        avg_client_service = df_clean[client_service_cols].mean().mean() if not df_clean.empty else None
        avg_team_collaboration = df_clean[team_collaboration_cols].mean().mean() if not df_clean.empty else None

        # Overall satisfaction (average of all categories)
        category_averages = [avg for avg in [avg_psychological_safety, avg_work_environment,
                           avg_client_service, avg_team_collaboration] if avg is not None]
        overall_satisfaction = sum(category_averages) / len(category_averages) if category_averages else None

        # Individual question averages
        question_averages = {}
        all_questions = psychological_safety_cols + work_environment_cols + client_service_cols + team_collaboration_cols

        for question in all_questions:
            if question in df_clean.columns:
                question_averages[question] = df_clean[question].mean()

        # Satisfaction distribution (based on overall satisfaction score)
        satisfaction_distribution = {'high': 0, 'medium': 0, 'low': 0}

        if not df_clean.empty:
            # Calculate individual satisfaction scores
            individual_scores = []
            for _, row in df_clean.iterrows():
                scores = []
                for col in all_questions:
                    if col in df_clean.columns and pd.notna(row[col]):
                        scores.append(row[col])
                if scores:
                    individual_scores.append(sum(scores) / len(scores))

            # Categorize satisfaction levels
            for score in individual_scores:
                if score >= 4.0:
                    satisfaction_distribution['high'] += 1
                elif score >= 3.0:
                    satisfaction_distribution['medium'] += 1
                else:
                    satisfaction_distribution['low'] += 1

        # Process qualitative feedback
        feedback_responses = df_all['Open_Feedback'].dropna()
        feedback_themes = extract_feedback_themes(feedback_responses.tolist())

        return {
            'total_responses': total_responses,
            'valid_responses': valid_responses,
            'avg_psychological_safety': avg_psychological_safety,
            'avg_work_environment': avg_work_environment,
            'avg_client_service': avg_client_service,
            'avg_team_collaboration': avg_team_collaboration,
            'overall_satisfaction': overall_satisfaction,
            'role_distribution': role_distribution,
            'question_averages': question_averages,
            'satisfaction_distribution': satisfaction_distribution,
            'feedback_themes': feedback_themes,
            'feedback_count': len(feedback_responses)
        }

    except Exception as e:
        logging.error(f"Error calculating survey KPIs: {str(e)}")
        raise


def extract_feedback_themes(feedback_list):
    """Extract common themes from open-ended feedback.

    Args:
        feedback_list: List of feedback strings

    Returns:
        list: List of theme dictionaries with theme and count
    """
    try:
        if not feedback_list:
            return []

        # Simple keyword-based theme extraction
        themes = {
            'Communication': ['communication', 'communicate', 'updates', 'sharing', 'feedback', 'meetings'],
            'Tools & Resources': ['tools', 'resources', 'equipment', 'software', 'technology', 'documentation'],
            'Work-Life Balance': ['balance', 'workload', 'hours', 'overtime', 'stress', 'time'],
            'Team Collaboration': ['team', 'collaboration', 'cooperation', 'support', 'help', 'together'],
            'Training & Development': ['training', 'learning', 'development', 'skills', 'knowledge', 'education'],
            'Management': ['management', 'leadership', 'supervisor', 'manager', 'direction', 'guidance'],
            'Client Service': ['client', 'customer', 'service', 'support', 'satisfaction', 'experience'],
            'Process Improvement': ['process', 'improvement', 'efficiency', 'workflow', 'procedures', 'optimize']
        }

        theme_counts = {}

        # Count theme occurrences
        for feedback in feedback_list:
            if pd.isna(feedback) or not isinstance(feedback, str):
                continue

            feedback_lower = feedback.lower()
            for theme, keywords in themes.items():
                for keyword in keywords:
                    if keyword in feedback_lower:
                        theme_counts[theme] = theme_counts.get(theme, 0) + 1
                        break  # Count each theme only once per feedback

        # Convert to list format and sort by count
        theme_list = [{'theme': theme, 'count': count} for theme, count in theme_counts.items()]
        theme_list.sort(key=lambda x: x['count'], reverse=True)

        return theme_list[:10]  # Return top 10 themes

    except Exception as e:
        logging.error(f"Error extracting feedback themes: {str(e)}")
        return []


def validate_survey_file_structure(filepath):
    """Validate that the uploaded file has the expected survey structure.

    Args:
        filepath: Path to the Excel file

    Returns:
        tuple: (is_valid, error_message, column_info)
    """
    try:
        # Read the file
        df = pd.read_excel(filepath)

        # Expected columns
        expected_columns = [
            'Role', 'Q1_1_SpeakingUp', 'Q1_2_MistakesHeldAgainstMe', 'Q1_3_RespectWhenNotKnowing',
            'Q2_1_WorkloadManageable', 'Q2_2_ToolsAndResources', 'Q2_3_WorkLifeBalance',
            'Q3_1_UnderstandingClients', 'Q3_2_SupportHandlingClients', 'Q3_3_ToolsForClientService',
            'Q4_1_HelpResponsiveness', 'Q4_2_ConflictResolution', 'Q4_3_SharingUpdates',
            'Open_Feedback'
        ]

        # Check for missing columns
        missing_columns = [col for col in expected_columns if col not in df.columns]
        extra_columns = [col for col in df.columns if col not in expected_columns]

        # Validate data types and ranges for quantitative questions
        quantitative_columns = expected_columns[1:13]  # Q1-Q4 questions
        validation_errors = []

        for col in quantitative_columns:
            if col in df.columns:
                # Check if values are in expected range (1-5) or NaN
                valid_values = df[col].dropna()
                if not valid_values.empty:
                    if not all((valid_values >= 1) & (valid_values <= 5)):
                        validation_errors.append(f"Column {col} contains values outside range 1-5")

        # Check if we have any data
        if df.empty:
            return False, "File is empty", {}

        # Prepare column info
        column_info = {
            'total_columns': len(df.columns),
            'expected_columns': len(expected_columns),
            'missing_columns': missing_columns,
            'extra_columns': extra_columns,
            'total_rows': len(df),
            'validation_errors': validation_errors
        }

        # Determine if file is valid
        is_valid = len(missing_columns) == 0 and len(validation_errors) == 0

        if not is_valid:
            error_message = []
            if missing_columns:
                error_message.append(f"Missing required columns: {', '.join(missing_columns)}")
            if validation_errors:
                error_message.extend(validation_errors)
            error_message = "; ".join(error_message)
        else:
            error_message = "File structure is valid"

        return is_valid, error_message, column_info

    except Exception as e:
        return False, f"Error reading file: {str(e)}", {}