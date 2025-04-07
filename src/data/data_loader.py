import pandas as pd
import xmltodict
import numpy as np

def load_xml_data(file_path):
    """Load and parse XML data into a pandas DataFrame."""
    with open(file_path, 'r', encoding='utf-8') as file:
        xml_data = xmltodict.parse(file.read())
    
    # Find the REPORT worksheet
    workbook = xml_data['ss:Workbook']
    worksheets = workbook['Worksheet']
    report_worksheet = None
    
    for worksheet in worksheets:
        if worksheet['@ss:Name'] == 'REPORT':
            report_worksheet = worksheet
            break
    
    if not report_worksheet:
        raise ValueError("REPORT worksheet not found")
    
    table = report_worksheet['Table']
    rows = table['Row']
    
    # Skip the first 4 rows (title and metadata)
    data_rows = rows[4:]
    
    # Extract headers (first row after metadata)
    headers = []
    header_cells = data_rows[0]['Cell']
    for cell in header_cells:
        if 'Data' in cell and '#text' in cell['Data']:
            headers.append(cell['Data']['#text'])
        else:
            headers.append(f'Column_{len(headers)}')
    
    # Process data rows
    data = []
    for row in data_rows[1:]:  # Skip header row
        if 'Cell' not in row:
            continue
            
        row_data = [''] * len(headers)  # Initialize with empty strings
        cells = row['Cell']
        if not isinstance(cells, list):
            cells = [cells]  # Handle single cell case
        
        for i, cell in enumerate(cells):
            if 'Data' in cell and '#text' in cell['Data']:
                # Get the index, accounting for merged cells
                if 'ss:Index' in cell:
                    idx = int(cell['ss:Index']) - 1
                else:
                    idx = i
                
                if idx < len(row_data):
                    row_data[idx] = cell['Data']['#text']
        
        # Only add rows that have some non-empty values
        if any(val != '' for val in row_data):
            data.append(row_data)
    
    df = pd.DataFrame(data, columns=headers)
    print(f"\nSample of data from {file_path}:")
    print(df.head())
    return df

def load_tariffs(file_path='data/tariffs.csv'):
    """Load tariff data from CSV file."""
    tariffs_df = pd.read_csv(file_path)
    # Convert to dictionary for easier lookup
    tariffs = dict(zip(tariffs_df['Country'], tariffs_df['Tariff Rate']))
    return tariffs 