"""
Module for parsing HTML-based Excel files.

This module provides functionality to parse HTML files that are formatted as Excel spreadsheets,
which is a common format used by web applications for exporting data.
"""

import pandas as pd
import re
import logging
from bs4 import BeautifulSoup
import os
import tempfile

def is_html_excel(file_path):
    """
    Check if the file is an HTML-based Excel file.
    
    Args:
        file_path (str): Path to the file to check
        
    Returns:
        bool: True if the file appears to be an HTML-based Excel file, False otherwise
    """
    try:
        # Read the first few lines of the file
        with open(file_path, 'rb') as f:
            header = f.read(1024).decode('utf-8', errors='ignore')
        
        # Check for HTML and Excel-related patterns
        html_patterns = [
            '<html', '<!DOCTYPE html', '<table', 
            'application/vnd.ms-excel', 'Excel', 'excel'
        ]
        
        return any(pattern.lower() in header.lower() for pattern in html_patterns)
    except Exception as e:
        logging.warning(f"Error checking if file is HTML Excel: {str(e)}")
        return False

def parse_html_excel(file_path):
    """
    Parse an HTML-based Excel file into a pandas DataFrame.
    
    Args:
        file_path (str): Path to the HTML Excel file
        
    Returns:
        pandas.DataFrame: DataFrame containing the parsed data
    """
    try:
        logging.info(f"Parsing HTML-based Excel file: {file_path}")
        
        # Read the HTML file
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find the main table - usually the largest one
        tables = soup.find_all('table')
        if not tables:
            logging.warning("No tables found in HTML file")
            return None
        
        # Find the table with the most rows (likely the main data table)
        main_table = max(tables, key=lambda t: len(t.find_all('tr')))
        
        # Extract table data
        rows = []
        for tr in main_table.find_all('tr'):
            row = []
            for td in tr.find_all(['td', 'th']):
                # Clean the cell text
                cell_text = td.get_text(strip=True)
                row.append(cell_text)
            if row:  # Skip empty rows
                rows.append(row)
        
        if not rows:
            logging.warning("No data rows found in HTML table")
            return None
        
        # Create DataFrame
        if len(rows) > 1:
            # Use the first row as header
            df = pd.DataFrame(rows[1:], columns=rows[0])
        else:
            df = pd.DataFrame(rows)
        
        logging.info(f"Successfully parsed HTML Excel file with {len(df)} rows and {len(df.columns)} columns")
        return df
    
    except Exception as e:
        logging.error(f"Error parsing HTML Excel file: {str(e)}")
        return None

def convert_html_to_excel(file_path):
    """
    Convert an HTML-based Excel file to a real Excel file.
    
    Args:
        file_path (str): Path to the HTML Excel file
        
    Returns:
        str: Path to the converted Excel file, or None if conversion failed
    """
    try:
        # Parse the HTML file
        df = parse_html_excel(file_path)
        if df is None:
            return None
        
        # Create a temporary file
        temp_dir = tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, "converted_excel.xlsx")
        
        # Save as Excel
        df.to_excel(temp_file, index=False)
        logging.info(f"Successfully converted HTML Excel to real Excel: {temp_file}")
        
        return temp_file
    
    except Exception as e:
        logging.error(f"Error converting HTML Excel to real Excel: {str(e)}")
        return None
